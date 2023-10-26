import sys
import optparse
from tqdm.auto import tqdm
import wandb
import os
from transformers import AutoTokenizer
from transformers.activations import ACT2FN
import torch
from typing import Tuple
from torch import LongTensor, Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from dataclasses import dataclass
import dataclasses
from typing import Optional

from rotary_embedding_torch import RotaryEmbedding
import mmh3
from bakadata import get_dl, get_tokenizer, batch_iterator, MiniBatch


def model_numel(m: nn.Module):
    return sum(p.numel() for p in m.parameters())


@dataclass
class BakaState:
    """ Normalized state of the layer """
    input: Optional[Tensor] = None
    output: Optional[Tensor] = None
    offset: int = 0
    past_predcessor_state: Optional["BakaState"] = None  # if this is layer L at current chunk C, this field points to layer (L-1) in chunk (C-1)
    k_cache: Optional[Tensor] = None
    v_cache: Optional[Tensor] = None
    attn_flip_state: bool = False

    @property
    def n_seq(self):
        return self.input.shape[1]

    @property
    def device(self):
        return self.input.device


@dataclass
class BakaConfig:
    dim_model: int = 1024
    dim_ff: int = 0
    dim_attn: int = 0
    n_layers: int = 12
    n_heads: int = 4
    n_vocab: int = 32000
    n_ctx: int = 512

    act_fn: str = "elephant"
    model_type = "baka_xl"

    def __post_init__(self):
        self.dim_ff = self.dim_ff or self.dim_model * 4
        self.dim_attn = self.dim_attn or self.dim_model
        assert self.dim_attn % self.n_heads == 0

    @property
    def dim_head(self):
        return self.dim_attn // self.n_heads


class BakaElephant(nn.Module):
    def __init__(self, d=8.0, a=1.0) -> None:
        super().__init__()
        self.d = d
        self.a = a

    def forward(self, state):
        state = 1 / (1 + (state / self.a).abs()**self.d)
        return state


BAKA_ACTS = {
    "elephant": BakaElephant
}


def make_activation_fn(s):
    if act := BAKA_ACTS.get(s):
        return act()
    return ACT2FN[s]


class BakaMLP(nn.Module):
    def __init__(self, config: BakaConfig) -> None:
        super().__init__()
        self.config = config
        self.act = make_activation_fn(self.config.act_fn)
        self.fc_gate = nn.Linear(config.dim_model, config.dim_ff, False)
        self.fc_up = nn.Linear(config.dim_model, config.dim_ff, False)
        self.fc_down = nn.Linear(config.dim_ff, config.dim_model, False)

    def forward(self, state: BakaState):
        gate = self.act(self.fc_gate(state.input))
        y = self.fc_up(state.input) * gate
        y = self.fc_down(y)
        return y


class BakaAttention(nn.Module):
    def __init__(self, config: BakaConfig) -> None:
        super().__init__()
        self.config = config
        self.q = nn.Linear(config.dim_model, config.dim_attn, False)
        self.k = nn.Linear(config.dim_model, config.dim_attn, False)
        self.v = nn.Linear(config.dim_model, config.dim_model, False)
        self.o = nn.Linear(config.dim_model, config.dim_model, False)
        self.rot = RotaryEmbedding(config.dim_head, use_xpos=False)

    def forward(self, state: BakaState):
        q, k, v = self.build_qkv(state)
        is_causal, attn_mask = self.build_attn_mask(state)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, attn_mask=attn_mask)
        y = rearrange(y, "b h t f -> b t (h f)")
        y = self.o(y)
        return y

    def build_attn_mask(self, state: BakaState):
        # Will not be required in future versions of pytorch, hopefully.
        # See https://github.com/pytorch/pytorch/issues/108108
        if state.past_predcessor_state is None:
            return True, None
        now_n_seq = state.n_seq
        past_n_seq = state.past_predcessor_state.n_seq

        # Build a matrix for both past and present
        # 1 1 1 1  1 1 1
        # 1 1 1 1  1 1 1
        # 1 1 1 1  1 1 1
        mask = torch.ones(now_n_seq, now_n_seq + past_n_seq, dtype=torch.bool, device=state.device)
        # by shifting diagonal we can cut from current chunk only
        mask = mask.tril_(past_n_seq)
        # 1 1 1 1  1 0 0
        # 1 1 1 1  1 1 0
        # 1 1 1 1  1 1 1
        return False, mask

    def build_qkv(self, state: BakaState):
        # project
        q = self.q(state.input)
        k = self.k(state.input)
        v = self.v(state.input)

        q = rearrange(q, "b t (h f) -> b h t f", h=self.config.n_heads)
        k = rearrange(k, "b t (h f) -> b h t f", h=self.config.n_heads)
        v = rearrange(v, "b t (h f) -> b h t f", h=self.config.n_heads)

        # Cache keys without rotating (temporary?)
        state.k_cache = k
        state.v_cache = v

        past_offset = current_offset = 0

        # restore the past
        if (past := state.past_predcessor_state):
            past_k = past.k_cache
            past_v = past.v_cache
            assert past_k is not None and past_v is not None
            k = torch.cat((past_k, k), -2)
            v = torch.cat((past_v, v), -2)
            current_offset = past.n_seq

        q = self.rot.rotate_queries_or_keys(q, -2, offset=current_offset)
        k = self.rot.rotate_queries_or_keys(k, -2, offset=past_offset)

        return q, k, v


class BakaLayer(nn.Module):
    def __init__(self, config: BakaConfig) -> None:
        super().__init__()
        self.config = config
        self.norm_in = nn.LayerNorm(config.dim_model)
        self.attn = BakaAttention(config)
        self.mlp = BakaMLP(config)

    def forward(self, state: BakaState):
        y = state.input
        assert y is not None
        state.input = self.norm_in(state.input)
        attn = self.attn(state)
        mlp = self.mlp(state)
        y = y + attn + mlp
        state.output = y
        return y


class BakaNet(nn.Module):
    def __init__(self, config: BakaConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([BakaLayer(config) for _ in range(config.n_layers)])

    def forward(self, input: Tensor, old_states: Optional[list[BakaState]] = None) -> Tuple[Tensor, list[BakaState]]:
        # TODO output_states flag

        n_batch, n_seq, n_dim = input.shape
        # Del unused vars to calm down my editor's LSP
        del n_batch
        del n_dim

        # Prepare placeholder for past state if nothing was provided
        if old_states is None:
            old_states = [None for _ in range(self.config.n_layers)]

        # Calculate the start position from the start position and length of the last chunk (if any)
        start_pos = 0
        if old_states[0]:
            start_pos = old_states[0].offset + old_states[0].n_seq

        # Prepare outputs for each chunk
        outputs = []

        flip_flop = False

        # Move in context left to right
        for pos in range(0, n_seq, self.config.n_ctx):
            # Prepare this step state
            states = [BakaState(input=None, offset=start_pos + pos, attn_flip_state=flip_flop) for _ in range(self.config.n_layers)]

            # pass input from the current context window
            states[0].input = input[:, pos:pos+self.config.n_ctx, :]

            # Move in the model top to bottom
            for i, layer in enumerate(self.layers):
                # Link current state to the past state
                states[i].past_predcessor_state = old_states[i-1] if i else None

                # Link output of the intermediate layer to input of the next intermediate layer
                if i:
                    states[i].input = states[i-1].output

                # Let the layer cook current state
                layer(states[i])

            # Last layer represents everything we know so far
            outputs.append(states[-1].output)

            # Shift current states into old states
            old_states = states

            # Detach older states
            for state in old_states:
                state.past_predcessor_state = None
            flip_flop = not flip_flop

        outputs = torch.cat(outputs, 1)
        return outputs, old_states


@dataclass
class BakaCausalLMOutput:
    logits: Tensor
    loss: Optional[Tensor] = None
    states: Optional[BakaState] = None


class BakaNetCausalLM(nn.Module):
    def __init__(self, config: BakaConfig) -> None:
        super().__init__()
        self.config = config
        self.model = BakaNet(config)
        self.vocab_in = nn.Embedding(config.n_vocab, config.dim_model)
        self.norm_out = nn.LayerNorm(config.dim_model)
        self.vocab_out = nn.Linear(config.dim_model, config.n_vocab, False)

        # HF compatibility hack
        self.model.get_input_embeddings = lambda: self.vocab_in

    def forward(self,
                input_ids: LongTensor,
                labels: Optional[LongTensor] = None,
                attention_mask="ignored",
                old_states: Optional[list[BakaState]] = None
                ):
        y = self.vocab_in(input_ids)
        y, states = self.model(y, old_states)
        y = self.norm_out(y)
        y = self.vocab_out(y)
        loss = None
        if labels is not None:
            y_pred = rearrange(y[:, :-1, :], 'b t f -> (b t) f')
            y_true = rearrange(labels[:, 1:], 'b t -> (b t)')
            loss = F.cross_entropy(y_pred, y_true)
        return BakaCausalLMOutput(loss=loss, logits=y, states=states)


WANDB_INITED = False


def wandb_log(**kwargs):
    global WANDB_INITED
    project = kwargs.pop("project")
    run_id = kwargs.pop("run_id")
    if not WANDB_INITED:
        WANDB_INITED = True

        wandb.init(
            project=project,
            id=run_id,
            resume=True,
            config={})

    wandb.log(kwargs)


def try_load(obj, path):
    if not os.path.exists(path):
        print(f"*** {path} doesn't exist ")
        return False
    print(f"Loading {path}")
    obj.load_state_dict(torch.load(path))
    return True


def make_model(tokenizer) -> Tuple[BakaConfig, BakaNetCausalLM]:
    cfg = BakaConfig(
        dim_model=768,
        dim_ff=3072,
        n_heads=12,
        n_layers=12,
        n_vocab=len(tokenizer))
    model = BakaNetCausalLM(cfg).bfloat16().cuda()
    return model


def gen_model_path(project, model):
    cfg = model.config
    numel = model_numel(model)
    cfg_hash = hex(mmh3.hash(str(cfg), signed=False))[2:]
    model_path = f"weights/baka_{project}_{numel}_{cfg_hash}.bin"
    return model_path


def propogate_past_states(past: list[BakaState], mb: MiniBatch):
    if not past:
        return
    keys = list(dataclasses.asdict(BakaState()).keys())

    for state in past:
        if not state:
            continue
        for k in keys:
            v = getattr(state, k)
            if not isinstance(v, torch.Tensor):
                continue
            v = mb.pass_batch_from_past_to_present(v)
            setattr(state, k, v)


def train_batch(model, tokenizer, batch, training_ctx_size, opt, clip, n_skip_first=0, write_log=None):
    past = None
    for mb in batch_iterator(batch, n_ctx=training_ctx_size, n_stride=training_ctx_size, tokenizer=tokenizer, n_skip_first=n_skip_first):
        past = propogate_past_states(past, mb)
        out = model(input_ids=mb.input_ids, labels=mb.labels, old_states=past)
        past = out.states
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        opt.zero_grad()
        if write_log:
            write_log(loss, mb)


def main():
    parser = optparse.OptionParser()
    parser.add_option("-p", "--project", dest="project",
                      help="set project name to PROJECT", metavar="PROJECT")
    parser.add_option("-r", "--run", dest="run_id",
                      help="set run id to RUN_ID", metavar="RUN_ID")
    parser.add_option("-w", "--wandb", dest="do_log", action="store_true",
                      help="enable WANDB log")
    parser.add_option("-l", "--load", dest="do_load", action="store_true",
                      help="load existing model")
    parser.add_option("-S", "--no-save", dest="do_save", action="store_false", default=True,
                      help="do not save the progress(default: save)")

    debug_args = None
    if sys.gettrace():
        print("Debug mode")
        debug_args = ["--project", "debug", "--run", "debug-01", "-S"]

    options, _ = parser.parse_args(debug_args)

    project: str = options.project
    run_id: str = options.run_id
    assert run_id and run_id[-1].isdigit()
    do_log = options.do_log
    do_load = options.do_load
    do_save = options.do_save or do_load

    assert project, "project name is required"
    assert run_id, "run id is required"
    tokenizer = get_tokenizer()
    dl = get_dl(tokenizer, batch_size=5)
    clip = 1.0
    #
    model = make_model(tokenizer)
    training_ctx_size = 2048
    opt = torch.optim.AdamW(model.parameters())

    # PATH
    model_path = gen_model_path(project, model)
    opt_path = model_path + ".opt"
    ###
    print(f"#parms: {model_numel(model)}")
    if do_load:
        try_load(model, model_path)
        try_load(opt, opt_path)

    def write_log(loss: Tensor, mb: MiniBatch):
        bar.set_description(f'L:{loss.item():.4f}, P:{mb.progress:%}x{mb.n_batch} (len:{mb.seqtotal})')
        if do_log:
            wandb_log(loss=loss.item(), project=f"baka3-{project}", run_id=run_id)

    for i_batch, batch in enumerate(bar := tqdm(dl)):
        train_batch(model, tokenizer, batch, training_ctx_size, opt, clip, write_log=write_log)
        train_batch(model, tokenizer, batch, training_ctx_size, opt, clip, n_skip_first=training_ctx_size//2, write_log=write_log)

        if do_save and i_batch and i_batch % 50 == 0:
            torch.save(model.state_dict(), model_path)
            torch.save(opt.state_dict(), opt_path)

    if do_save:
        torch.save(model.state_dict(), model_path)
        torch.save(opt.state_dict(), opt_path)
    print("DONE")


if __name__ == "__main__":
    main()
