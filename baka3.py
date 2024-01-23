import sys
import numpy as np
import optparse
from tqdm.auto import tqdm
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

def model_numel(m: nn.Module, grad_only=False):
    return sum(p.numel() * (p.requires_grad or not grad_only)
               for p in m.parameters())

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
    past_state: Optional["BakaState"] = None  # if this is layer L at current chunk C, this field points to layer L in chunk (C-1)
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
    dim_model_start: int = 256
    dim_model_increment: int = 128
    dim_ff_mult: float = 4.0
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
    model_type = "baka_fanout"
    use_flash_attn: bool = True
    def __post_init__(self):
        assert self.dim_model_increment > 0


    def dim_attn(self, layer_num: int):
        return self.dim_model(layer_num)

    def dim_ff(self, layer_num: int):        
        #return int(self.dim_model(layer_num) * self.dim_ff_mult)
        del layer_num
        return int(self.dim_model(self.n_layers) * self.dim_ff_mult)

    def dim_model(self, layer_num:int):
        return self.dim_model_start + layer_num * self.dim_model_increment

    def dim_head(self, layer_num:int):
        return self.dim_attn(layer_num) // self.n_heads

def make_config(tokenizer):
    cfg = BakaConfig(
        dim_model_start=384,
        dim_model_increment=48,
        dim_ff_mult=4.0,
        n_heads=12,
        n_layers=12,
        n_vocab=len(tokenizer))
    return cfg


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
    def __init__(self, config: BakaConfig, layer_num: int) -> None:
        super().__init__()
        self.config = config
        self.dim_in = config.dim_model(layer_num)
        self.dim_ff = config.dim_ff(layer_num)
        self.dim_out = config.dim_model(layer_num+1)
        self.num_layer = layer_num
        self.act = make_activation_fn(self.config.act_fn)

        self.fc_gate = nn.Linear(self.dim_in, self.dim_ff, False)
        self.fc_up = nn.Linear(self.dim_in, self.dim_ff, False)
        self.fc_down = nn.Linear(self.dim_ff, self.dim_out, False)

    def forward(self, state: BakaState|Tensor):
        input = state.input if isinstance(state, BakaState) else state # TODO: is it cleaner to fix MLP to accept tensor only? meh
        gate = self.act(self.fc_gate(input))
        y = self.fc_up(input) * gate
        y = self.fc_down(y)
        return y


class BakaAttention(nn.Module):
    def __init__(self, config: BakaConfig, layer_num: int) -> None:
        super().__init__()
        self.config = config
        self.dim_in = config.dim_model(layer_num)
        self.dim_out = config.dim_model(layer_num+1)
        self.dim_attn = config.dim_attn(layer_num)
        self.num_layer = layer_num

        self.q = nn.Linear(self.dim_in, self.dim_attn, False)
        self.k = nn.Linear(self.dim_in, self.dim_attn, False)
        self.v = nn.Linear(self.dim_attn, self.dim_attn, False)
        self.o = nn.Linear(self.dim_attn, self.dim_out, False)
        self.rot = RotaryEmbedding(config.dim_head(layer_num), use_xpos=False)

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
        if state.past_state is None:
            # No RMT here, which is not really ideal as RMT[-2] will not attend to following RMT[-1],
            # but it happens only once per sequence
            return True, None
        now_n_seq = state.n_seq
        past_n_seq = state.past_state.n_seq

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
        # TODO: do I need predcessor here?
        if (past := state.past_state):
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

class BakaZeroPad(nn.Module):
    # Control, to do: remove once better upscaler found
    def __init__(self, dim_in:int, dim_out:int) -> None:
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

    def forward(self, x: Tensor):
        n_batch, n_seq = x.shape[0], x.shape[1]
        n_pad = self.dim_out - self.dim_in
        pad = torch.zeros(n_batch, n_seq, n_pad, dtype=x.dtype, device=x.device)
        y = torch.cat((x, pad), -1)
        return y

class BakaLinearUpscaler(nn.Module):
    # upscale_missing=False ==> results unknown
    # upscale_missing=True ==> Garbage, to do: remove
    def __init__(self, dim_in:int, dim_out:int, upscale_missing_elements_only=False) -> None:
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.upscale_missing_elements_only=upscale_missing_elements_only

        delta = self.dim_out - self.dim_in
        if upscale_missing_elements_only:
            assert dim_out > dim_in
            self.fc = nn.Linear(dim_in, delta, False)
        else:
            self.fc = nn.Linear(dim_in, dim_out, False)
        self.fc.weight.data *= 1e-4

    def forward(self, x: Tensor):
        y = self.fc(x)
        if self.upscale_missing_elements_only:
            y = torch.cat((x, y), -1)
        else:
            y = x + y # todo config?
        return y

class BakaLinearPerHeadUpscaler(nn.Module):
    def __init__(self, n_heads:int, dim_in:int, dim_out:int) -> None:
        # should I pass bakaconfig for consistency?
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_heads = n_heads 

        self.fc = nn.Linear(dim_in, dim_out - dim_in, False)
        self.fc.weight.data *= 1e-4

    def forward(self, x: Tensor):
        y = self.fc(x)
        y = rearrange(y, "b t (h c) -> b t h c", h = self.n_heads)
        x = rearrange(x, "b t (h c) -> b t h c", h = self.n_heads)
        y = torch.cat((x, y), -1)
        y = rearrange(y, "b t h c -> b t (h c)")
        return y

class BakaZeroPadPerHeadUpscaler(nn.Module):
    def __init__(self, n_heads:int, dim_in:int, dim_out:int) -> None:
        # should I pass bakaconfig for consistency?
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_heads = n_heads 


    def forward(self, x: Tensor):
        n_batch, n_seq = x.shape[0], x.shape[1]
        y = torch.zeros(n_batch, n_seq, self.dim_out - self.dim_in, dtype=x.dtype, device=x.device)
        y = rearrange(y, "b t (h c) -> b t h c", h = self.n_heads)
        x = rearrange(x, "b t (h c) -> b t h c", h = self.n_heads)
        y = torch.cat((x, y), -1)
        y = rearrange(y, "b t h c -> b t (h c)")
        return y

class BakaLayer(nn.Module):
    def __init__(self, config: BakaConfig, layer_num: int) -> None:
        super().__init__()
        self.config = config
        self.dim_in = config.dim_model(layer_num)
        self.dim_out = config.dim_model(layer_num+1)
        self.norm_in = nn.LayerNorm(self.dim_in)
        self.attn = BakaAttention(config, layer_num)
        self.mlp = BakaMLP(config, layer_num)
        self.upscaler = BakaZeroPadPerHeadUpscaler(config.n_heads, self.dim_in, self.dim_out)
        self.enabled = True

    def enabled_(self, enabled=True):
        self.enabled = enabled
        return self

    def forward(self, state: BakaState):
        # If layer is disabled, only upscaler is active
        x0 = state.input
        assert x0 is not None
        xp = self.upscaler(x0)

        if not self.enabled:
            xp = self.upscaler(state.input)
            state.output = xp
            return xp

        state.input = self.norm_in(state.input)
        attn = self.attn(state)
        mlp = self.mlp(state)
        y = xp + attn + mlp
        state.output = y
        return y


class BakaRMT(nn.Module):
    def __init__(self, config: BakaConfig):
        super().__init__()
        self.config = config
        self.rmt_tokens = nn.Parameter(torch.randn(config.n_rmt_tokens, config.dim_model(0)))
        # TODO: remap

    def inject(self, current_start: BakaState, last_end: Optional[BakaState]) -> BakaRMTState:
        assert current_start.input is not None

        if last_end is not None:
            assert last_end.output is not None
            assert last_end.rmt is not None
            lhs = rhs = last_end.output[:, -last_end.rmt.rhs:, :self.config.dim_model(0)]
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
        self.pause_emb = nn.Parameter(torch.randn(config.dim_model(0)))

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
        self.layers = nn.ModuleList([BakaLayer(config, n) for n in range(config.n_layers)])
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
                states[i].past_state = old_states[i]
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
                state.past_state = None

        outputs = torch.cat(outputs, 1)
        return outputs, old_states or [] # type: ignore

    def stacking_training(self, n_start: int, n_train: int):
        for n, layer in enumerate(self.layers):
            if n < n_start:
                print(f"Freezing layer {n}")
                layer.enabled_(True)
                layer.requires_grad_(False)
            elif n < n_start + n_train:
                print(f"Focusing on layer {n}")
                layer.enabled_(True)
                layer.requires_grad_(True)
            else:
                print(f"Disabling layer {n}")
                layer.requires_grad_(False)
                layer.enabled_(False)




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
        self.vocab_in = nn.Embedding(config.n_vocab, config.dim_model(0))
        self.norm_out = nn.LayerNorm(config.dim_model(config.n_layers))
        self.vocab_out = nn.Linear(config.dim_model(config.n_layers), config.n_vocab, False)

        # HF compatibility hack
        self.model.get_input_embeddings = lambda: self.vocab_in # type: ignore

    def stacking_training(self, n_start: int, n_train: int):
        self.model.stacking_training(n_start, n_train)
        if n_start > 0:
            self.vocab_in.requires_grad_(False)

    def forward(self,
                input_ids: LongTensor,
                labels: Optional[LongTensor] = None,
                attention_mask="ignored",
                old_states: Optional[list[BakaState]] = None
                ):
        del attention_mask
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


def try_load(obj, path):
    if not os.path.exists(path):
        print(f"*** {path} doesn't exist ")
        return False
    print(f"Loading {path}")
    obj.load_state_dict(torch.load(path))
    return True


def make_model(tokenizer) -> BakaNetCausalLM:
    cfg = make_config(tokenizer)
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
    parser.add_option("-w", "--write-log", dest="do_log", action="store_true",
                      help="enable log")
    parser.add_option("-l", "--load", dest="do_load", action="store_true",
                      help="load existing model")
    parser.add_option("-b", "--batch-size", dest="batch_size", default=5, type="int",
                      help="do not save the progress(default: save)")
    parser.add_option("-s", "--skip", dest="skip", type="int", metavar="N", default=0,
                      help="skip N batches")
    parser.add_option("-S", "--no-save", dest="do_save", action="store_false", default=True,
                      help="do not save the progress(default: save)")
    parser.add_option("-d", "--detach", metavar="N", help="stop gradient after N steps", type="int", default=8)
    parser.add_option("-k", "--keep", metavar="A,N", help="train N layers after layer A", type="str", default=None)

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
    keep: str = options.keep

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

    if keep:
        n_start, n_train = map(int, keep.split(","))
        model.stacking_training(n_start, n_train)
    else:
        model.requires_grad_(True)
    print(f"#trainable: {hl_color}{model_numel(model, True)}{reset_color}")


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
        torch.save(model.state_dict(), model_path+"."+run_id)
        torch.save(opt.state_dict(), opt_path)
    print("DONE")


if __name__ == "__main__":
    #torch.use_deterministic_algorithms(True)
    main()
