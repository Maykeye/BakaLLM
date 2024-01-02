#!/usr/bin/env python
import sys
import optparse
from functools import partial
import torch
import math
import os
import uuid
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm.auto import tqdm
from typing import Optional
import sqlite3
import mmh3
from dataclasses import dataclass
from bakadata import get_dl, get_tokenizer
import baka3

MODELS_PATH = os.environ.get("MODELS_PATH", "~/models/")
DEVICE = "cuda"


def model_numel(x):
    return sum(p.numel() for p in x.parameters())


@dataclass
class TestContext:
    model: AutoModelForCausalLM
    main_model: torch.nn.Module
    tokenizer: AutoTokenizer
    n_ctx: int
    n_batch: int = 1
    config_id: str = ""


def impl_load_transformers(model_id, dtype=torch.float, n_ctx=2048, trust_remote_code=False):
    transformers_models_path = os.path.expanduser(MODELS_PATH)
    model_path = transformers_models_path + model_id
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=trust_remote_code).to(device=DEVICE, dtype=dtype)
    if model.config.model_type == "gpt_neox":
        base_model = model.gpt_neox
    elif model.config.model_type in ("gpt2", "btlm"):
        base_model = model.transformer
    elif model.config.model_type in ("llama", "mistral"):
        base_model = model.model
    else:
        raise ValueError(f"Unknown model {model.config.model_type}")
    return TestContext(model=model, main_model=base_model, tokenizer=tokenizer, n_ctx=n_ctx)


def load_bakanet(project, n_ctx=1024*1024, forced_path=None):
    tokenizer = get_tokenizer()
    model = baka3.make_model(tokenizer)
    state_dict_path = forced_path or baka3.gen_model_path(project, model)
    print(f"loading {state_dict_path}")
    assert Path(state_dict_path).exists(), f"path not found {state_dict_path}"
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)
    model = model.to(device=DEVICE)
    return TestContext(model=model, main_model=model.model, tokenizer=tokenizer, n_ctx=n_ctx)


def load_baka_pythia():
    config_path = MODELS_PATH + "pythia-160m-deduped"
    model_config = AutoConfig.from_pretrained(os.path.expanduser(config_path))
    model = AutoModelForCausalLM.from_config(model_config)
    window_size = 1024
    state_dict_path = f"weights/p2wiki103_p160_{model_numel(model)}_{window_size}.bin"
    print(f"loading {state_dict_path}")
    assert Path(state_dict_path).exists(), f"path not found {state_dict_path}"
    tokenizer = get_tokenizer()
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)
    model = model.to(device=DEVICE)
    return TestContext(model=model, main_model=model.gpt_neox, tokenizer=tokenizer, n_ctx=window_size)


@torch.no_grad()
def run(ctx: TestContext):
    dl = get_dl(ctx.tokenizer, batch_size=ctx.n_batch, split="validation")
    losses = []
    nans = 0
    for b in tqdm(dl):
        n_seq = len(b['input_ids'][0])
        for i in range(0, n_seq, ctx.n_ctx//2):
            inputs = {k: v[:, i:i+ctx.n_ctx].to(device=DEVICE) for k, v in b.items()}
            n_seq = inputs["input_ids"].shape[1]
            labels = inputs['input_ids'].clone()
            if i:
                labels[:, :ctx.n_ctx//2] = -100
                if (labels == -100).all():
                    break
            inputs["labels"] = labels
            loss = ctx.model(**inputs).loss
            if loss.isnan():
                nans += 1
            else:
                losses.append(loss.item())
    return torch.mean(torch.tensor(losses)).item(), nans


def write_hashed_text(cursor: sqlite3.Cursor, table_name: str, column_name: str, string: str):
    hash = mmh3.hash(string)
    l = (cursor
         .execute(f"SELECT id FROM {table_name} WHERE hash = ? and {column_name} = ?", (hash, string))
         .fetchall())

    if l:
        return l[0][0]

    record_id = uuid.uuid4().bytes
    cursor.execute(f"INSERT INTO {table_name}(id, hash, {column_name}) VALUES(?, ?, ?)",
                   (record_id, hash, string))
    return record_id


def record(ctx: TestContext, loss, nans):
    model = ctx.model
    n_ctx = ctx.n_ctx
    record_id = uuid.uuid4().bytes
    config_id = ctx.config_id
    con = sqlite3.connect("results/results.sqlite3")
    n_param = sum(p.numel() for p in model.parameters())
    main_model = ctx.main_model
    n_input_emb_params = sum(p.numel() for p in main_model.get_input_embeddings().parameters())
    benchmark = "wikipedia103"
    benchmark_split = "valid"
    note = ctx.note
    with con as cursor:
        model_architecture_id = write_hashed_text(cursor, "ModelArchitecture", "architecture", str(model))

        cursor.execute("""INSERT INTO Result(id, config_id, model_type, 
                       n_params, n_input_emb_params, 
                       benchmark, benchmark_split,
                       nans, n_ctx, loss, ppl,
                       model_architecture_id, note) 
    VALUES (?,?,?,  ?,?,  ?,?, ?,?,?,?, ?,?)""", (
            record_id, config_id, model.config.model_type,
            n_param, n_input_emb_params,
            benchmark, benchmark_split,
            nans, n_ctx, loss, math.exp(loss),
            model_architecture_id, note))

        con.commit()
    return record_id


def main():
    global DEVICE
    print("Loading the model")
    parser = optparse.OptionParser()
    parser.add_option("-m", "--model", dest="model", help="load model")
    parser.add_option("-p", "--model-path", dest="model_path", help="force to load bakanet state_dict from the given file")
    parser.add_option("-n", "--n_ctx", dest="n_ctx", type="int", help="context size")
    parser.add_option("-t", "--textnote", dest="note", help="note to myself")
    parser.add_option("-d", "--device", dest="device", help="device", default=DEVICE)
    options, _ = parser.parse_args()
    DEVICE=options.device

    loaders = {
        "pythia-14m": partial(impl_load_transformers, "pythia-14m"),
        "pythia-31m": partial(impl_load_transformers, "pythia-31m"),
        "pythia-70m": partial(impl_load_transformers, "pythia-70m-deduped"),
        "pythia-160m": partial(impl_load_transformers, "pythia-160m-deduped"),
        "pythia-1b": partial(impl_load_transformers, "pythia-1b-deduped"),
        "cerebras-111m": partial(impl_load_transformers, "Cerebras-GPT-111M"),
        "cerebras-256m": partial(impl_load_transformers, "Cerebras-GPT-256M"),
        "gpt2": partial(impl_load_transformers, "gpt2", n_ctx=1024),
        "open-llama-3b": partial(impl_load_transformers, "open_llama_3b", dtype=torch.bfloat16),
        "open-llama-7b": partial(impl_load_transformers, "open_llama_7b", dtype=torch.bfloat16, n_ctx=1024),
        "btlm-3b": partial(impl_load_transformers, "btlm-3b-8k-base", dtype=torch.bfloat16, trust_remote_code=True),
        "mistral-7b": partial(impl_load_transformers, "Mistral-7B-v0.1", dtype=torch.bfloat16, n_ctx=1024),
        ####
        "baka-pythia-160m": load_baka_pythia,
        "baka-xl": partial(load_bakanet, "baka-xl", forced_path=options.model_path),
    }
    if not options.model:
        print(f"Use the following loaders: {list(loaders.keys())}")
        quit()
    if options.n_ctx:
        print("n_ctx: ", type(options.n_ctx))

    ctx = loaders[options.model]()
    ctx.config_id = options.model
    ctx.note = options.note
    if options.n_ctx:
        ctx.n_ctx = options.n_ctx
    print("Loaded")

    aloss, nans = run(ctx)
    print(f"AVGLOSS: {aloss:.5f}, PPL: {math.exp(aloss):.5f} NANS:{nans}, NCTX: {ctx.n_ctx}")
    record_id = record(ctx, aloss, nans)
    print(f"Recorded {record_id}")


if __name__ == "__main__":
    main()
