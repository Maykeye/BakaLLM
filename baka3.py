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
from typing import Optional

from rotary_embedding_torch import RotaryEmbedding
import mmh3
from bakadata import get_dl, get_tokenizer, batch_iterator


def model_numel(m: nn.Module):
    return sum(p.numel() for p in m.parameters())


@dataclass
class BakaState:
    """ Normalized state of the layer """
    input: Optional[Tensor] = None
    output: Optional[Tensor] = None
    offset = 0
    past_predcessor_state: Optional["BakaState"] = None  # if this is layer L at current chunk C,  this field points to layer (L-1) in chunk (C-1)


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
    model_type = "baka_elephant"

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
        del state
        return True, None

    def build_qkv(self, state: BakaState):
        # project
        q = self.q(state.input)
        k = self.k(state.input)
        v = self.v(state.input)

        q = rearrange(q, "b t (h f) -> b h t f", h=self.config.n_heads)
        k = rearrange(k, "b t (h f) -> b h t f", h=self.config.n_heads)
        v = rearrange(v, "b t (h f) -> b h t f", h=self.config.n_heads)

        # pos embeds
        q = self.rot.rotate_queries_or_keys(q, -2, offset=state.offset)
        k = self.rot.rotate_queries_or_keys(k, -2, offset=state.offset)
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

    def forward(self, input: Tensor):
        n_batch, n_seq, n_dim = input.shape
        del n_batch
        del n_dim
        old_states = [None for _ in range(self.config.n_layers)]
        outputs = []
        # FIXME: offset is not being used
        for pos in range(0, n_seq, self.config.n_ctx):
            states = [BakaState(input=None) for _ in range(self.config.n_layers)]
            states[0].input = input[:, pos:pos+self.config.n_ctx, :]
            for i, layer in enumerate(self.layers):
                states[i].past_predcessor_state = old_states[i-1] if i else None
                if i:
                    states[i].input = states[i-1].output
                layer(states[i])
            outputs.append(states[-1].output)
            old_states = states

            # Detach older states
            for state in old_states:
                state.past_predcessor_state = None

        outputs = torch.cat(outputs, 1)
        return outputs


@dataclass
class BakaCausalLMOutput:
    logits: Tensor
    loss: Optional[Tensor] = None


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

    def forward(self, input_ids: LongTensor, labels: Optional[LongTensor] = None, attention_mask="ignored"):
        y = self.vocab_in(input_ids)
        y = self.model(y)
        y = self.norm_out(y)
        y = self.vocab_out(y)
        loss = None
        if labels is not None:
            y_pred = rearrange(y[:, :-1, :], 'b t f -> (b t) f')
            y_true = rearrange(labels[:, 1:], 'b t -> (b t)')
            loss = F.cross_entropy(y_pred, y_true)
        return BakaCausalLMOutput(loss=loss, logits=y)


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
    options, _ = parser.parse_args()

    project: str = options.project
    run_id: str = options.run_id
    do_log = options.do_log
    do_load = options.do_load

    assert project, "project name is required"
    assert run_id, "run id is required"
    tokenizer = get_tokenizer()
    dl = get_dl(tokenizer, batch_size=6)
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

    for i_batch, batch in enumerate(bar := tqdm(dl)):
        # TODO: once memory instlaled - either increase stride to n_ctx or don't use memory across mini-batches
        for mb in batch_iterator(batch,
                                 n_ctx=training_ctx_size,
                                 n_stride=training_ctx_size // 2,
                                 tokenizer=tokenizer):
            out = model(input_ids=mb.input_ids, labels=mb.labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            opt.zero_grad()
            bar.set_description(f'L:{loss.item():.4f}, P:{mb.progress:%}x{mb.n_batch} (len:{mb.seqtotal})')
            if do_log:
                wandb_log(loss=loss.item(), project=f"baka3-{project}", run_id=run_id)
        if i_batch and i_batch % 50 == 0:
            torch.save(model.state_dict(), model_path)
            torch.save(opt.state_dict(), opt_path)
    torch.save(model.state_dict(), model_path)
    torch.save(opt.state_dict(), opt_path)
    print("DONE")


if __name__ == "__main__":
    main()
