import sys
import numpy as np
import optparse
from tqdm.auto import tqdm
import wandb
import os
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
from float_logger import flog
from transformers import set_seed

import flash_attn

def my_log(**kwargs):
    flog(**kwargs)
    wandb_log(**kwargs)

def model_numel(m: nn.Module):
    return sum(p.numel() for p in m.parameters())

@dataclass
class BakaRMTState:
    lhs: int = 0
    rhs: int = 0

@dataclass
class BakaState:
    """ Normalized state of the layer """
    input: Optional[Tensor] = None
    output: Optional[Tensor] = None
    offset: int = 0
    past_state: Optional["BakaState"] = None  # if this is layer L at current chunk C, this field points to layer (L-1) in chunk (C-1)
    past_predcessor_state: Optional["BakaState"] = None  # if this is layer L at current chunk C, this field points to layer (L-1) in chunk (C-1)
    k_cache: Optional[Tensor] = None
    v_cache: Optional[Tensor] = None
    pauses_pos: Optional[list[int]] = None
    rmt: Optional[BakaRMTState] = None
    @property
    def n_seq(self):
        assert self.input is not None
        return self.input.shape[1]
    @property
    def n_batch(self):
        assert self.input is not None
        return self.input.shape[0]

    @property
    def device(self):
        assert self.input is not None
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
    n_pauses: int = 8
    is_pause_pos_random: bool = False
    n_rmt_tokens: int = 16
    n_rmt_margins: int = 0
    rmt_solidifier: str = "none" # none, glu, mlp, gelephant
    act_fn: str = "elephant"
    model_type = "baka_rmt"
    use_flash_attn: bool = True

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

    def forward(self, state: BakaState|Tensor):
        input = state.input if isinstance(state, BakaState) else state # TODO: is it cleaner to fix MLP to accept tensor only? meh
        gate = self.act(self.fc_gate(input))
        y = self.fc_up(input) * gate
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
        if self.config.use_flash_attn:
            # NB: Flash attention ignores RMT mask
            y = flash_attn.flash_attn_func(q, k, v, causal=True) 
        else:
            is_causal, attn_mask = self.build_attn_mask(state)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, attn_mask=attn_mask)


        arrangement = "b t h f" if self.config.use_flash_attn else "b h t f"
        arrangement += " -> b t (h f)"

        y = rearrange(y, arrangement)
        y = self.o(y)
        return y

    def build_attn_mask(self, state: BakaState):
        # Will not be required in future versions of pytorch, hopefully.
        # See https://github.com/pytorch/pytorch/issues/108108
        if state.past_predcessor_state is None:
            # No RMT here, which is not really ideal as RMT[-2] will not attend to following RMT[-1],
            # but it happens only once per sequence
            return True, None
        now_n_seq = state.n_seq
        past_n_seq = state.past_predcessor_state.n_seq

        # Build a matrix for both past and present
        # 1 1 1 1  1 1 1
        # 1 1 1 1  1 1  
        # 1 1 1 1  1 1 1
        mask = torch.ones(now_n_seq, now_n_seq + past_n_seq, dtype=torch.bool, device=state.device)
        # by shifting diagonal we can cut from current chunk only
        mask = mask.tril_(past_n_seq)
        # 1 1 1 1  1 0 0
        # 1 1 1 1  1 1 0
        # 1 1 1 1  1 1 1
        if state.rmt:
            if state.rmt.rhs:
                # Write RMT memory applies to everything
                mask[-state.rmt.rhs:] = 1
            if state.rmt.lhs:
                # Read RMT memory applies to itself, but not current block unlike the paper["Additionally, we allow all
                # memory tokens in the read/write block to access all other tokens in the same block. "]:
                # If model learns top copy first N tokens into first N tokens of RMT-Read, during PPL calculation of these first N tokens, it can easily
                # cheat and read them, which will be a disaster during the inference
                # TODO: can be optimized by fusing it into tril_ above
                mask[:state.rmt.lhs, past_n_seq:past_n_seq+state.rmt.lhs] = 1

        return False, mask

    def build_qkv(self, state: BakaState):
        # project
        q = self.q(state.input)
        k = self.k(state.input)
        v = self.v(state.input)

        arrangement = "b t (h f) -> "
        arrangement += "b t h f" if self.config.use_flash_attn else "b h t f"

        q = rearrange(q, arrangement, h=self.config.n_heads)
        k = rearrange(k, arrangement, h=self.config.n_heads)
        v = rearrange(v, arrangement, h=self.config.n_heads)

        # Cache keys without rotating (temporary?)
        state.k_cache = k
        state.v_cache = v

        past_offset = current_offset = 0

        # restore the past
        if (past := state.past_predcessor_state):
            past_k = past.k_cache
            past_v = past.v_cache
            assert past_k is not None and past_v is not None
            seq_dim = -3 if self.config.use_flash_attn else -3
            k = torch.cat((past_k, k), seq_dim)
            v = torch.cat((past_v, v), seq_dim)
            current_offset = past.n_seq
        seq_dim = -3 if self.config.use_flash_attn else -2
        q = self.rot.rotate_queries_or_keys(q, seq_dim, offset=current_offset)
        k = self.rot.rotate_queries_or_keys(k, seq_dim, offset=past_offset)

        return q, k, v


def make_post_normalized(n_dim: int, base: nn.Module):
    return nn.Sequential(base, nn.LayerNorm(n_dim))


class BakaLayer(nn.Module):
    def __init__(self, config: BakaConfig) -> None:
        super().__init__()
        self.config = config
        self.norm_in = nn.LayerNorm(config.dim_model)
        self.attn = BakaAttention(config)
        self.mlp = BakaMLP(config)

    def forward(self, state: BakaState):
        x0 = state.input
        assert x0 is not None
        state.input = self.norm_in(state.input)
        attn = self.attn(state)
        mlp = self.mlp(state)
        y = x0 + attn + mlp
        state.output = y
        return y


class BakaRMT(nn.Module):
    def __init__(self, config: BakaConfig):
        super().__init__()
        self.config = config
        self.rmt_tokens = nn.Parameter(torch.randn(config.n_rmt_tokens, config.dim_model))

    def inject(self, current_start: BakaState, last_end: Optional[BakaState]) -> BakaRMTState:
        assert current_start.input is not None

        if last_end is not None:
            assert last_end.output is not None
            assert last_end.rmt is not None
            lhs = rhs = last_end.output[:, -last_end.rmt.rhs:]
            current_start.input = torch.cat((lhs, current_start.input, rhs), 1)
            return BakaRMTState(lhs=lhs.shape[1], rhs=rhs.shape[1])

        rhs = self.rmt_tokens.repeat(current_start.n_batch, 1, 1)
        current_start.input = torch.cat((current_start.input, rhs), 1)
        return BakaRMTState(lhs=0, rhs=rhs.shape[1])

    def detach(self, x: Tensor, state: BakaRMTState):
        if state.lhs:
            x = x[:, state.lhs:]
        if state.rhs:
            x = x[:, :-state.rhs]
        return x


class BakaPause(nn.Module):
    def __init__(self, config: BakaConfig) -> None:
        super().__init__()
        self.config = config
        self.pause_emb = nn.Parameter(torch.randn(config.dim_model))

    def inject(self, state:BakaState):
        assert state.input is not None
        def rand_pos(n) -> int:
            return int(torch.randint(1, n+1, (1,)).item())

        # first pos is fixed to zero to emulate BOS
        if self.config.is_pause_pos_random:
            state.pauses_pos = [0] + [rand_pos(state.n_seq + i) for i in range(1, self.config.n_pauses)]
        else:
            step = self.config.n_ctx // (self.config.n_pauses)
            state.pauses_pos = [step * i for i in range(self.config.n_pauses) if step*i <= state.n_seq]

        pause_emb = self.pause_emb.repeat(state.n_batch, 1, 1)
        x = state.input
        for pos in state.pauses_pos:
            x = torch.cat((x[:, :pos], pause_emb, x[:, pos:]), 1)
        state.input = x

    def removed(self, x: Tensor, pauses_pos: list[int]):
        for pos in reversed(pauses_pos):
            x = torch.cat((x[:, :pos], x[:, pos+1:]), 1)
        return x


class BakaNet(nn.Module):
    def __init__(self, config: BakaConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([BakaLayer(config) for _ in range(config.n_layers)])
        self.pause = BakaPause(config)
        self.rmt = BakaRMT(config)

    def forward(self,
                input: Tensor,
                old_states: Optional[list[Optional[BakaState]]] = None # type: ignore
        ) -> Tuple[Tensor, list[BakaState]]:

        n_batch, n_seq, n_dim = input.shape
        # Del unused vars to calm down my editor's LSP
        del n_batch
        del n_dim

        # Prepare placeholder for past state if nothing was provided
        if old_states is None:
            old_states = [None for _ in range(self.config.n_layers)] #type: ignore


        # Calculate the start position from the start position and length of the last chunk (if any)
        start_pos = 0
        if old_states[0] is not None:
            start_pos = old_states[0].offset + old_states[0].n_seq

        # Prepare outputs for each chunk
        outputs = []
        # Move in context left to right
        for pos in range(0, n_seq, self.config.n_ctx):
            # Prepare this step state
            states = [BakaState(input=None, offset=start_pos + pos) for _ in range(self.config.n_layers)]

            # pass input from the current context window
            states[0].input = input[:, pos:pos+self.config.n_ctx, :]
            self.pause.inject(states[0])
            rmt_state = self.rmt.inject(states[0], old_states[-1])
            states[0].rmt = rmt_state

            # Move in the model top to bottom
            for i, layer in enumerate(self.layers):
                # Link current state to the past state
                states[i].past_predcessor_state = old_states[i-1] if i else None
                states[i].pauses_pos = states[0].pauses_pos
                states[i].rmt = states[0].rmt

                # Link output of the intermediate layer to input of the next intermediate layer
                if i:
                    states[i].input = states[i-1].output

                # Let the layer cook current state
                layer(states[i])

            assert states[-1].output is not None
            assert states[-1].pauses_pos is not None
            # Last layer represents everything we know so far, which is more than necessary
            # To get output, useable for loss calc, we need to remove pauses, RMT, etc
            output = self.rmt.detach(states[-1].output, rmt_state)
            output = self.pause.removed(output, states[-1].pauses_pos)
            outputs.append(output)

            # Shift current states into old states
            old_states: list[BakaState] = states #type: ignore

            # Detach older states completely
            for state in old_states:
                state.past_predcessor_state = None

        outputs = torch.cat(outputs, 1)
        return outputs, old_states or [] # type: ignore


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
        self.model.get_input_embeddings = lambda: self.vocab_in # type: ignore

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


def make_model(tokenizer) -> BakaNetCausalLM:
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


def propogate_past_states(past: Optional[list[BakaState]], mb: MiniBatch, detach=False):
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
            if detach:
                v = v.detach()
            setattr(state, k, v)

def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def autoclip_gradient(model: nn.Module, grad_history: list[float], clip_percentile: float=10.0):
    obs_grad_norm = get_grad_norm(model)
    grad_history.append(obs_grad_norm)
    clip_value = np.percentile(grad_history, clip_percentile)
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value) #type: ignore

AUTOCLIP = "autoclip"
DYNAMIC_AUTOCLIP="dynamic"
AUTOCLIP_HISTORY=[]
def train_batch(model, tokenizer, batch, training_ctx_size, opt, clip, n_skip_first=0, detach_at = 0, write_log=None):
    past = None

    for i, mb in enumerate(batch_iterator(batch,
                                          n_ctx=training_ctx_size,
                                          n_stride=training_ctx_size,
                                          tokenizer=tokenizer,
                                          n_skip_first=n_skip_first)):
        should_detach = detach_at != 0 and i % detach_at == 0
        past = propogate_past_states(past, mb, detach=should_detach)
        out = model(input_ids=mb.input_ids, labels=mb.labels, old_states=past)
        past = out.states
        loss = out.loss
        loss.backward()
        if clip is not None:
            if clip == AUTOCLIP:
                autoclip_gradient(model, AUTOCLIP_HISTORY)
            elif clip == DYNAMIC_AUTOCLIP:
                clip_alpha = 0.5
                clip_beta = 2 / torch.e
                step_clip = clip_alpha * (mb.progress) + clip_beta * (1-mb.progress)
                torch.nn.utils.clip_grad_norm_(model.parameters(), step_clip) #type: ignore
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip) #type: ignore
        opt.step()
        opt.zero_grad()
        if write_log:
            write_log(loss, mb)


def main():
    set_seed(9)
    parser = optparse.OptionParser()
    parser.add_option("-p", "--project", dest="project",
                      help="set project name to PROJECT", metavar="PROJECT")
    parser.add_option("-r", "--run", dest="run_id",
                      help="set run id to RUN_ID", metavar="RUN_ID")
    parser.add_option("-w", "--wandb", dest="do_log", action="store_true",
                      help="enable WANDB log")
    parser.add_option("-l", "--load", dest="do_load", action="store_true",
                      help="load existing model")
    parser.add_option("-b", "--batch-size", dest="batch_size", default=5, type="int",
                      help="do not save the progress(default: save)")
    parser.add_option("-s", "--skip", dest="skip", type="int", metavar="N", default=0,
                      help="skip N batches")
    parser.add_option("-S", "--no-save", dest="do_save", action="store_false", default=True,
                      help="do not save the progress(default: save)")
    parser.add_option("-d", "--detach", metavar="N", help="stop gradient after N steps", type="int", default=8)

    debug_args = None
    if sys.gettrace():
        print("Debug mode")
        debug_args = ["--project", "debug", "--run", "debug-01", "-S"]

    options, _ = parser.parse_args(debug_args)

    project: str = options.project
    run_id: str = options.run_id
    assert run_id and run_id[-1].isdigit()
    do_log: bool = options.do_log
    do_load: bool = options.do_load
    do_save: bool = options.do_save or do_load
    batch_size: int = options.batch_size
    detach_at: int = options.detach

    assert project, "project name is required"
    assert run_id, "run id is required"
    tokenizer = get_tokenizer()
    dl = get_dl(tokenizer, batch_size=batch_size, n_skip_batches=options.skip)

    clip = 0.5

    model = make_model(tokenizer)
    training_ctx_size = 2048
    n_ctx = model.config.n_ctx
    opt = torch.optim.AdamW(model.parameters())

    # PATH
    model_path = gen_model_path(project, model)
    opt_path = model_path + ".opt"
    ###
    hl_color = "\x1b[1;32m"
    reset_color = "\x1b[0m"
    print(f"#parms: {hl_color}{model_numel(model)}{reset_color}; path: {hl_color}{model_path}{reset_color}")

    if do_load:
        try_load(model, model_path)
        try_load(opt, opt_path)



    def write_log(loss: Tensor, mb: MiniBatch):
        bar.set_description(f'L:{loss.item():.4f}, P:{mb.progress:%}x{mb.n_batch} (len:{mb.seqtotal})')
        if do_log:
            my_log(loss=loss.item(), project=f"baka3-{project}", run_id=run_id)

    for i_batch, batch in enumerate(bar := tqdm(dl)):
        train_batch(model, tokenizer, batch, training_ctx_size, opt, clip,
                    n_skip_first=0, detach_at=detach_at, write_log=write_log)
        train_batch(model, tokenizer, batch, training_ctx_size, opt, clip,
                    n_skip_first=n_ctx//2, detach_at=detach_at, write_log=write_log)

        if do_save and i_batch and i_batch % 50 == 0:
            torch.save(model.state_dict(), model_path)
            torch.save(opt.state_dict(), opt_path)

    if do_save:
        torch.save(model.state_dict(), model_path)
        torch.save(opt.state_dict(), opt_path)
    print("DONE")


if __name__ == "__main__":
    #torch.use_deterministic_algorithms(True)
    main()
