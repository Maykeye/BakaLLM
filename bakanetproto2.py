import sys
from maykio import batch_iterator_dict
import maykds
from transformers import set_seed
from typing import Optional, Tuple, Literal
from torch import Tensor, FloatTensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding
from einops import rearrange
from transformers.modeling_outputs import ModelOutput
from transformers.activations import get_activation
from tqdm.auto import tqdm
from pathlib import Path
from torch.cuda.amp import GradScaler
from maykio import io_setup, wandb_log
from torch.utils.data import DataLoader
from copy import deepcopy
from optparse import OptionParser
import dataclasses
from transformers import AutoConfig, AutoModelForCausalLM

from scipy.fftpack import next_fast_len
from torch.fft import rfft, irfft


def append_dims(x, num_dims):
    # https://github.com/lucidrains/Mega-pytorch/blob/main/mega_pytorch/mega_pytorch.py
    if num_dims <= 0:
        return x
    return x.view(*x.shape, *((1,) * num_dims))


def conv1d_fft(x, weights, dim=-2, weight_dim=-1):
    # https://github.com/lucidrains/Mega-pytorch/blob/main/mega_pytorch/mega_pytorch.py
    # O(N log(N)) 1d convolution using some fourier trick

    assert weight_dim >= dim

    N = x.shape[dim]
    M = weights.shape[weight_dim]

    fast_len = next_fast_len(N + M - 1)

    f_x = rfft(x.float(), n=fast_len, dim=dim)
    f_weight = rfft(weights.float(), n=fast_len, dim=weight_dim)
    # The FFT of a real signal is Hermitian-symmetric, ``X[i] = conj(X[-i])`` so
    # the output contains only the positive frequencies below the Nyquist frequency.
    # ^^^ 😮😮😮 ^^^
    f_v_weight = f_x * append_dims(f_weight.conj(), weight_dim - dim)
    out = irfft(f_v_weight, fast_len, dim=dim)
    out = out.roll(-1, dims=(dim,))

    indices = torch.arange(start=fast_len - N, end=fast_len, dtype=torch.long, device=x.device)
    out = out.index_select(dim, indices)
    return out.to(dtype=x.dtype)


@dataclasses.dataclass
class BakaLayerCache:
    input: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor

    def detach(self) -> "BakaLayerCache":
        def do_detach(x):
            return x.detach() if x is not None else None
        return BakaLayerCache(
            do_detach(self.input),
            do_detach(self.key),
            do_detach(self.value))

    @property
    def n_batch(self):
        return self.key.shape[0]

    @property
    def n_seq(self):
        return self.key.shape[-2]

    def prepend_kv_to(self, k, v):
        k = torch.cat((self.key, k), -2)
        v = torch.cat((self.value, v), -2)
        return k, v


@dataclasses.dataclass
class BakaNetModelOutput(ModelOutput):
    last_hidden_state: Tensor
    model_cache: Optional[list[BakaLayerCache]] = None


@dataclasses.dataclass
class BakaCausalLMOutput(ModelOutput):
    logits: Tensor
    loss: Optional[Tensor] = None
    model_cache: Optional[list[BakaLayerCache]] = None


class BakaNetConfig:
    def __init__(
        self,
        hidden_dim=768,
        num_layers=8,
        num_heads=1,
        num_memory_tokens=32,
        act_fn="relu",
        intermediate_size=None,
        window_size=512,
        vocab_size=32000,
        query_key_dim=None,
        is_causal=True
    ) -> None:
        self.num_layers = num_layers
        self.query_key_dim = query_key_dim or hidden_dim  # dimensionality of all heads combined
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_size or hidden_dim * 4
        self.num_memory_tokens = num_memory_tokens
        self.num_heads = num_heads
        self.act_fn = act_fn
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.is_causal = is_causal
        self.output_cache = False
        self.attn_dropout = 0.1
        self.mlp_dropout = 0.1
        self.mega_dropout = 0.1
        self.n_attention_last = None  # use attentiono on these tokens only. For internal usage
        self.model_type = "bakanet2_mega_qk_v_mismatch"

    def clone(self):
        return deepcopy(self)


class BakaSigmoidGate(nn.Module):
    def __init__(self, hidden_state) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden_state, hidden_state, False)

    def forward(self, x):
        gate = self.fc(x).sigmoid()
        return gate * x


class BakaAttentionQKV(nn.Module):
    def __init__(self, config: BakaNetConfig) -> None:
        super().__init__()
        self.config = config
        self.q = nn.Linear(config.hidden_dim, config.query_key_dim, False)
        self.k = nn.Linear(config.hidden_dim, config.query_key_dim, False)
        self.v = nn.Linear(config.hidden_dim, config.hidden_dim, False)
        self.num_heads = config.num_heads
        self.head_dim = config.query_key_dim // self.num_heads
        self.negator = nn.Parameter(torch.tensor(1.0))
        n = 2
        self.q_norm = nn.LayerNorm(config.query_key_dim * n)
        self.k_norm = nn.LayerNorm(config.query_key_dim * n)
        assert self.head_dim * self.num_heads == config.query_key_dim
        self.pos_emb = RotaryEmbedding(self.head_dim, use_xpos=False)

    def forward(self, q_state, k_state, v_state, offset):
        q = self.q(q_state)
        k = self.k(k_state)
        v = self.v(v_state)

        q = torch.cat((q, self.negator - q), -1)
        k = torch.cat((k, self.negator - k), -1)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.config.n_attention_last is not None:
            q = q[:, -self.config.n_attention_last:]

        q = rearrange(q, "b t (h f) -> b h t f", h=self.num_heads)
        k = rearrange(k, "b t (h f) -> b h t f", h=self.num_heads)
        v = rearrange(v, "b t (h f) -> b h t f", h=self.num_heads)

        q = self.pos_emb.rotate_queries_or_keys(q, -2, offset)
        k = self.pos_emb.rotate_queries_or_keys(k, -2, offset)
        return q, k, v


class BakaAttention(nn.Module):

    def __init__(self, config: BakaNetConfig) -> None:
        super().__init__()
        self.config = config
        self.qkv = BakaAttentionQKV(config=config)
        self.past_k_gate = BakaSigmoidGate(config.hidden_dim)
        self.past_v_gate = BakaSigmoidGate(config.hidden_dim)
        self.output_gate = BakaSigmoidGate(config.hidden_dim)
        self.num_heads = self.qkv.num_heads

    @staticmethod
    def build_attention_mask(n_old_ctx, n_query, device):
        # Build mask like
        #
        # F F F F F Q 0 0 0
        # F F F F F Q Q 0 0
        # F F F F F Q Q Q 0
        # F F F F F Q Q Q Q
        #
        # where F = 1: context tokens
        #       Q = 1: query tokens
        lhs = torch.ones(n_query, n_old_ctx, dtype=torch.bool, device=device)
        rhs = torch.ones(n_query, n_query, dtype=torch.bool, device=device).tril()
        mask = torch.cat((lhs, rhs), 1)
        return mask

    def forward(
        self,
        qk_state: FloatTensor,
        v_state: FloatTensor,
        last_cache: Optional[BakaLayerCache] = None,
        offset=0
    ):
        n_batch, n_seq, n_feature = qk_state.shape
        q, k, v = self.qkv(qk_state, qk_state, v_state, offset)
        if self.config.n_attention_last is not None:
            assert last_cache is None, "LC for partial query NYI"

        kv = BakaLayerCache(input=None, key=k, value=v)  # input will be filled late
        mask = None
        if last_cache is not None:
            assert self.config.is_causal, "LC for NC is NYI"
            past_k, past_v = last_cache.key, last_cache.value
            past_k = self.propogate_kv_cache(past_k, self.past_k_gate)
            past_v = self.propogate_kv_cache(past_v, self.past_v_gate)

            k = torch.cat((past_k, k), -2)
            v = torch.cat((past_v, v), -2)
            mask = self.build_attention_mask(n_old_ctx=last_cache.n_seq, n_query=q.shape[-2], device=q.device)
            mask = mask.repeat(n_batch, self.num_heads, 1, 1)

        dropout_p = self.config.attn_dropout if self.training else 0.0
        attn = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=self.config.is_causal and mask is None,
            dropout_p=dropout_p,
            attn_mask=mask)
        y = rearrange(attn, "b h t f -> b t (h f)")
        if self.config.n_attention_last is not None:
            y = torch.cat((qk_state[:, :-self.config.n_attention_last], y), 1)

        y = self.output_gate(y)
        return y, kv

    def propogate_kv_cache(self, kv, kv_gate):
        kv_shape = kv.shape
        kv = kv.view(-1, self.config.hidden_dim)
        kv = kv_gate(kv)
        kv = kv.view(*kv_shape)
        return kv


class BakaMLP(nn.Module):
    def __init__(self, config: BakaNetConfig) -> None:
        super().__init__()
        self.config = config
        self.fc_up = nn.Linear(config.hidden_dim, config.intermediate_dim, False)
        self.fc_gate = nn.Linear(config.hidden_dim, config.intermediate_dim, False)
        self.act_fn = get_activation(config.act_fn)
        self.fc_down = nn.Linear(config.intermediate_dim, config.hidden_dim, False)

    def forward(self, inputs):
        x = inputs
        if self.config.n_attention_last:
            x = x[:, -self.config.n_attention_last:]
        up = self.fc_up(x)
        gate = self.act_fn(self.fc_gate(x))
        y = self.fc_down(up * gate)
        if self.config.n_attention_last:
            y = torch.cat((inputs[:, :-self.config.n_attention_last], y), 1)
        return y


class BakaMega(nn.Module):
    def __init__(self, config: BakaNetConfig) -> None:
        super().__init__()
        self.config = config

        self.dampeners = nn.Parameter(
            torch.tensor([[0.9999], [0.9899]]).repeat_interleave(config.hidden_dim, 1))

    def apply_learned_ema_with_damping(self, x):
        # https://github.com/lucidrains/Mega-pytorch/blob/main/mega_pytorch/mega_pytorch.py
        # The torch as of time of writing has memory leak trouble
        #  https://github.com/pytorch/pytorch/issues/94893
        #

        n_batch, n_seq, n_feature = x.shape
        alphas = self.dampeners[0].sigmoid()
        dampen_factors = self.dampeners[1].sigmoid()
        reversed_powers = torch.arange(n_seq - 1, -1, -1, device=x.device)
        K = alphas * (((1 - alphas) * dampen_factors) ** rearrange(reversed_powers, '... l -> ... l 1'))
        y = conv1d_fft(x, K, dim=-2, weight_dim=-2)
        return y

    def forward(self, orig_inputs: Tensor, last_cache: Optional[BakaLayerCache] = None):
        n_seq = orig_inputs.shape[1]
        all_inputs = orig_inputs
        last_inputs = None if last_cache is None else last_cache.input

        if last_inputs is not None:
            all_inputs = torch.cat((last_inputs, orig_inputs), 1)
        all_inputs = self.apply_learned_ema_with_damping(all_inputs)
        y = all_inputs[:, -n_seq:, :]
        return y


class BakaMemoryInjector(nn.Module):
    def __init__(self, config: BakaNetConfig) -> None:
        super().__init__()
        self.n_memory_tokens = config.num_memory_tokens
        self.memory_tokens = nn.Parameter(torch.randn(2, self.n_memory_tokens, config.hidden_dim))
        self.sep_tokens = nn.Parameter(torch.randn(2, config.hidden_dim))
        solidifier_config = config.clone()
        solidifier_config.is_causal = False  # No need for causal attention anymore
        solidifier_config.n_attention_last = self.n_memory_tokens
        self.memory_solidifer = BakaBlock(solidifier_config)

    def forward(self, state, prefix):
        if not self.n_memory_tokens:
            return state
        n_batch = state.shape[0]
        if prefix is None:
            prefix = self.memory_tokens[0].repeat(n_batch, 1, 1)
        prefix_sep = self.sep_tokens[0].repeat(n_batch, 1, 1)
        suffix_sep = self.sep_tokens[1].repeat(n_batch, 1, 1)
        suffix = self.memory_tokens[1].repeat(n_batch, 1, 1)
        state = torch.cat((prefix, prefix_sep, state, suffix_sep, suffix), 1)
        return state

    def detach_memory(self, state):
        if not self.n_memory_tokens:
            return state, None
        # idea of solidifier is to make output of the last layer to be more approachable by first layer
        # as right now it's "aimed" to satisfy lm_head(as it follows logic of actual logits)
        solidified = self.memory_solidifer(state)
        suffix = solidified[0][:, -self.n_memory_tokens:]
        state = state[:, self.n_memory_tokens+1:]
        state = state[:, :-(self.n_memory_tokens+1)]
        return state, suffix


class BakaBlock(nn.Module):
    def __init__(self, config: BakaNetConfig) -> None:
        super().__init__()
        self.config = config
        self.norm_input = nn.LayerNorm(config.hidden_dim)
        self.attn = BakaAttention(config)
        self.mlp = BakaMLP(config)
        self.mlp_attn_gate = nn.Linear(config.hidden_dim, config.hidden_dim, False)
        self.mlp_attn_norm = nn.LayerNorm(config.hidden_dim)
        self.mega = BakaMega(config)
        if self.mega is not None:
            self.norm_mega = nn.LayerNorm(config.hidden_dim)

        self.mega_gate = nn.Linear(config.hidden_dim, config.hidden_dim)

    def forward(
            self,
            input_state: Tensor,
            last_cache: Optional[BakaLayerCache] = None,
            offset=0
    ) -> Tuple[Tensor, BakaLayerCache]:

        # For informational purposes on how hidden state is supposed to be
        n_batch, n_seq, n_feature = input_state.shape
        if self.mega is not None:
            hidden_state = self.mega(input_state, last_cache=last_cache)
            mega_state = self.norm_mega(hidden_state)
        else:
            mega_state = self.norm_input(hidden_state)

        # Normalize
        hidden_state = self.norm_input(input_state)

        # Step forward consists of two substeps done in parallel (wrt to input data):
        # * Attention
        attn, cache = self.attn(hidden_state, mega_state, last_cache=last_cache, offset=offset)
        # * MLP
        mlp = self.mlp(hidden_state)
        mlp = mlp * self.mlp_attn_gate(self.mlp_attn_norm(attn)).sigmoid()

        # Result is added together
        # result = hidden_state + attn + mlp
        mega_state = mega_state * F.sigmoid(self.mega_gate(mega_state))
        result = input_state + attn + mlp + mega_state

        return result, cache


class BakaEmbedding(nn.Module):
    def __init__(self, n_vocab, n_dim) -> None:
        super().__init__()
        self.n_vocab = n_vocab
        self.n_dim = n_dim
        self.embed = nn.Embedding(n_vocab, n_dim)

    def forward(self, x: Tensor):
        assert (x >= 0).all()
        assert (x < self.n_vocab).all()
        return self.embed(x)


class BakaNet(nn.Module):
    def __init__(self, config: BakaNetConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(BakaBlock(config) for _ in range(config.num_layers))
        self.embed_in = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.memory_injector = BakaMemoryInjector(config)

    def get_input_embeddings(self):
        return self.embed_in

    def parse_inputs_embeds(self, input_ids, inputs_embeds):
        if input_ids is not None:
            inputs_embeds = self.embed_in(input_ids)
        else:
            assert inputs_embeds is not None, "Exactly one of input_ids or input_embds must be passed"
        return inputs_embeds

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        last_caches: Optional[list[BakaLayerCache]] = None,
        output_cache=None,
        offset=0
    ) -> BakaNetModelOutput:
        if output_cache is None:
            output_cache = self.config.output_cache
        inputs_embeds = self.parse_inputs_embeds(input_ids, inputs_embeds)
        n_batch, n_seq, n_feature = inputs_embeds.shape
        if last_caches:
            assert n_batch == last_caches[0].n_batch, "Past hidden state must have the same batch dimensionality"

        result = []
        if last_caches:
            history = [x for x in last_caches]
        else:
            history = [None for _ in self.layers]

        pass_grad = True  # TODO: N steps
        prefix = None
        kv: Optional[BakaLayerCache] = None
        for pos in range(0, n_seq, self.config.window_size):
            state = inputs_embeds[:, pos:pos+self.config.window_size, :]
            state = self.memory_injector(state, prefix)
            for i, layer in enumerate(self.layers):
                state, kv = layer(state, last_cache=history[i], offset=pos+offset)
                kv.input = state
                assert kv.key is not None
                assert kv.value is not None
                history[i] = kv if pass_grad else kv.detach()
                pass_grad = not pass_grad
            state, prefix = self.memory_injector.detach_memory(state)

            result.append(state)
        result = torch.cat(result, 1)

        return BakaNetModelOutput(last_hidden_state=result, model_cache=history)

    def numel(self):
        return sum(p.numel() for p in self.parameters())


class BakaNetForCausalLM(nn.Module):
    def __init__(self, config: BakaNetConfig) -> None:
        super().__init__()
        self.config = config
        self.model = BakaNet(config)
        self.lm_norm = nn.LayerNorm(config.hidden_dim)
        self.embed_out = nn.Linear(config.hidden_dim, config.vocab_size, False)

    def forward(
            self,
            input_ids,
            labels=None,
            attention_mask=None,  # ignored
            last_caches: Optional[list[BakaLayerCache]] = None,
            output_cache: Optional[bool] = None,
            offset=0
    ) -> BakaCausalLMOutput:
        if output_cache is None:
            output_cache = self.config.output_cache
        out: BakaNetModelOutput = self.model(input_ids, last_caches=last_caches, output_cache=output_cache, offset=offset)
        y = self.lm_norm(out.last_hidden_state)
        y = self.embed_out(y)
        loss = None
        if labels is not None:
            y_pred = rearrange(y[:, :-1, :], 'b t f -> (b t) f')
            y_true = rearrange(labels[:, 1:], 'b t -> (b t)')
            loss = F.cross_entropy(y_pred, y_true)
        return BakaCausalLMOutput(loss=loss, logits=y, model_cache=out.model_cache)

    def numel(self):
        return sum(p.numel() for p in self.parameters())


@torch.no_grad()
def dirty_eval(model, input_ids, labels, losses, bar=None):
    model.eval()
    loss = model(input_ids=input_ids, labels=labels).loss
    losses.append(loss.item())
    model.train()
    return losses


@torch.no_grad()
def gentext(model, prompt, max_tokens):
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    for i in range(max_tokens):
        logits = model(input_ids=inputs).logits[0, -1]
        out = F.softmax(logits, -1)
        new_token_id = torch.multinomial(out, 1)
        inputs = F.pad(inputs, (0, 1), value=new_token_id.item())
    return tokenizer.decode(inputs.ravel())


tokenizer = maykds.pythia_tokenizer()
config_ts = BakaNetConfig(hidden_dim=320,
                          num_heads=1,
                          vocab_size=len(tokenizer), window_size=512)
config_w103 = BakaNetConfig(hidden_dim=768,
                            intermediate_size=2048,
                            num_heads=1,
                            num_layers=12,
                            window_size=512,
                            vocab_size=len(tokenizer))


@dataclasses.dataclass
class TrainConfig:
    # loaders
    do_wiki = False
    do_tiny = False
    do_pg19 = False
    do_p160 = False
    # model setting
    do_bfloat = False
    do_log = False
    do_amp = True
    # train setting
    do_load = False
    steps_to_deval = 5000
    model_config: BakaNetConfig = None
    save_filename: str = ""
    dataloader: DataLoader = None
    skip = 0
    max_steps = None
    window_size = 512
    big_window_size = 512 * 4
    retain_steps = 1


def parse_argv() -> TrainConfig:
    # TODO: optparse the minute we need second option beside SKIP
    argv = sys.argv
    cfg = TrainConfig()
    parse_opts(cfg)
    if not (cfg.do_pg19 or cfg.do_wiki or cfg.do_tiny or cfg.do_p160):
        print("Assuming tiny dataset")
        cfg.do_tiny = True
    if cfg.do_wiki or cfg.do_pg19 or cfg.do_p160:
        cfg.steps_to_deval = 500000
        cfg.save_filename = "weights/p2wiki103tiny.bin"
        cfg.model_config = config_w103
        cfg.do_bfloat = cfg.do_log = True
        cfg.do_amp = False
        if cfg.do_wiki or cfg.do_p160:
            cfg.dataloader = maykds.wikitext103_dl(tokenizer=tokenizer, batch_size=3)
        elif cfg.do_pg19:
            cfg.dataloader = maykds.pg19_dl(tokenizer=tokenizer, batch_size=4)
        if cfg.do_p160:
            cfg.model_config = AutoConfig.from_pretrained(Path("~/models/pythia-160m-deduped").expanduser())
            cfg.big_window_size = 1024
            cfg.window_size = 1024
            cfg.save_filename = "weights/p2wiki103_p160.bin"
    elif cfg.do_tiny:
        cfg.model_config = config_ts
        cfg.save_filename = "weights/p2tiny.bin"
        cfg.dataloader = maykds.tiny_stories_dl(tokenizer=tokenizer, batch_size=6)
        cfg.retain_steps = 1
    else:
        raise ValueError("No dataset was chosen")

    return cfg


def propogate_caches_to_current_mb(mb, past: Optional[list[BakaLayerCache]]):
    if not past:
        return past
    return [BakaLayerCache(
        input=mb.pass_batch_from_past_to_present(x.input),
        key=mb.pass_batch_from_past_to_present(x.key),
        value=mb.pass_batch_from_past_to_present(x.value)
    ) for x in past]


def parse_opts(cfg: TrainConfig):
    parser = OptionParser()
    parser.add_option("-s", "--skip", dest="skip", help="skip SKIP steps", metavar="SKIP", type="int")
    parser.add_option("-n", "--max", dest="n_steps", help="do N steps at most", metavar="N", type="int")
    parser.add_option("-c", "--cfg", dest="cfg", help="config: tinystory[default], pg19, wiki103, wiki103p160", type="choice",
                      choices=["tinystories", "pg19", "wiki103", "p160"])
    parser.add_option("-l", "--load", dest="do_load", action="store_true")

    values, args = parser.parse_args()
    cfg.skip = values.skip
    cfg.max_steps = values.n_steps
    cfg.do_pg19 = values.cfg == "pg19"
    cfg.do_wiki = values.cfg == "wiki103"
    cfg.do_tiny = values.cfg == "tinystories"
    cfg.do_p160 = values.cfg == "p160"
    cfg.do_load = values.do_load
    assert values.cfg is None or (cfg.do_pg19 or cfg.do_wiki or cfg.do_tiny or cfg.do_p160)
    if cfg.skip:
        assert cfg.do_load, "Skipping requires load"


def finish_batch_training(model, scaler, opt, retain_steps):
    if not retain_steps:
        return 0
    scaler.step(opt)
    scaler.update()
    opt.zero_grad()
    return 0


def run(model: BakaNetForCausalLM, cfg: TrainConfig, e: int):
    NL = "\n"
    eval_scores_buf = []
    skip = cfg.skip
    sample = "(None)"  # gentext("Once upon a time", 15)
    steps_to_deval = cfg.steps_to_deval
    scaler = GradScaler(enabled=cfg.do_amp)

    opt = torch.optim.AdamW(model.parameters())
    retain_steps = 0

    opt_path = cfg.save_filename + ".opt"
    if cfg.do_load and Path(opt_path).exists():
        opt_state = torch.load(opt_path)
        opt.load_state_dict(opt_state)
        print(f"*** Loading optimizer state {opt_path}")

    steps = 0
    for i_batch, batch in enumerate(bar := tqdm(cfg.dataloader)):
        if e == 0 and skip is not None and skip > 0:
            bar.set_description("FFWD")
            skip -= 1
            continue
        steps += 1
        if cfg.max_steps and steps == cfg.max_steps:
            print("Step limit reached. Finishing")
            break
        input_ids, labels = batch["input_ids"].cuda(), batch["labels"].cuda()
        steps_to_deval -= 1
        if steps_to_deval <= -1000:
            print("DE:", torch.tensor(dirty_eval(model, input_ids, labels, eval_scores_buf)).mean().item())
            quit()
        if steps_to_deval <= 0:
            if retain_steps:  # flush
                retain_steps = finish_batch_training(model, scaler, opt, retain_steps)

            bar.set_description('EVAL')
            dirty_eval(model, input_ids, labels, eval_scores_buf)
            continue

        last_caches = None
        for mb in batch_iterator_dict(batch, cfg.big_window_size, tokenizer=tokenizer):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=cfg.do_amp):
                # If batch was cut off, cut off part that corresponds to past that was removed
                last_caches = propogate_caches_to_current_mb(mb, last_caches)
                out: BakaCausalLMOutput
                if isinstance(model, BakaNetForCausalLM):
                    out = model(input_ids=mb.input_ids, labels=mb.labels, last_caches=last_caches, output_cache=True)
                else:
                    out = model(input_ids=mb.input_ids, labels=mb.labels)
                loss = out.loss
                if isinstance(model, BakaNetForCausalLM):
                    last_caches = [x.detach() for x in out.model_cache]
                b_loss = loss / cfg.retain_steps
            retain_steps += 1
            scaler.scale(b_loss).backward(retain_graph=retain_steps < cfg.retain_steps)

            if retain_steps == cfg.retain_steps:
                bar.set_description(f'L:{loss.item():.4f}, P:{mb.progress:%}x{mb.n_batch} S:{sample.replace(f"{NL}"," ")}')
                retain_steps = finish_batch_training(model, scaler, opt, retain_steps)

            if cfg.do_log:
                wandb_log(loss=loss.item())
        if i_batch and i_batch % 50 == 0:
            torch.save(model.state_dict(), cfg.save_filename)
            torch.save(opt.state_dict(), opt_path)
            sample = gentext(model, "Once upon a time", 15)
    torch.save(model.state_dict(), cfg.save_filename)
    torch.save(opt.state_dict(), opt_path)


def model_numel(m: nn.Module):
    return sum(p.numel() for p in m.parameters())


def main():
    set_seed(123)
    cfg = parse_argv()

    if isinstance(cfg.model_config, BakaNetConfig):
        model = BakaNetForCausalLM(cfg.model_config)
    else:
        model = AutoModelForCausalLM.from_config(cfg.model_config)
    if cfg.do_bfloat:
        model = model.bfloat16()
    model = model.cuda()
    print(f"#parms {model_numel(model)}")
    cfg.save_filename = cfg.save_filename.replace(".bin", f"_{model_numel(model)}_{cfg.window_size}.bin")
    if cfg.do_load and Path(cfg.save_filename).exists():
        model.load_state_dict(torch.load(cfg.save_filename))
        print(f"*** Loading {cfg.save_filename},  skipping {cfg.skip} steps")
    else:
        print(f"{cfg.save_filename} not found")
    if cfg.save_filename:
        io_setup("./weights", project_name="pythia-160", run_id="run3")
    run(model, cfg, 0)
    quit(0)


if __name__ == "__main__":
    main()
