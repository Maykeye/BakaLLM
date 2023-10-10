#!/usr/bin/env python
import torch
import math
import os
import uuid
import datasets
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, AutoConfig
from tqdm.auto import tqdm
from mayknetproto2 import BakaNetForCausalLM, config_w103
from maykds import pythia_tokenizer
from typing import Optional
import sqlite3
import mmh3
from dataclasses import dataclass
device = "cuda"
batch_size = 1
source_code = None


def model_numel(x):
    return sum(p.numel() for p in x.parameters())


@dataclass
class TestContext:
    model: AutoModelForCausalLM
    main_model: torch.nn.Module
    tokenizer: AutoTokenizer
    n_ctx: int
    source: Optional[str] = None


def impl_load_transformers(model_id, dtype=torch.float, n_ctx=2048, trust_remote_code=False):
    transformers_models_path = os.path.expanduser("~/models/")
    model_path = transformers_models_path + model_id
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=trust_remote_code).to(device=device, dtype=dtype)
    if model.config.model_type == "gpt_neox":
        base_model = model.gpt_neox
    elif model.config.model_type in ("gpt2", "btlm"):
        base_model = model.transformer
    elif model.config.model_type in ("llama", "mistral"):
        base_model = model.model
    else:
        raise ValueError(f"Unknown model {model.config.model_type}")
    return TestContext(model=model, main_model=base_model, tokenizer=tokenizer, n_ctx=n_ctx)


def load_bakanet():
    global source_code
    model = BakaNetForCausalLM(config_w103).cuda()
    state_dict_path = f"weights/p2wiki103tiny_{model.numel()}_{model.config.window_size}.bin"
    print(f"loading {state_dict_path}")
    assert Path(state_dict_path).exists(), f"path not found {state_dict_path}"
    tokenizer = pythia_tokenizer()
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)
    model = model.to(device=device)
    source_code = Path("./bakanetproto2.py").read_text()
    return TestContext(model=model, main_model=model.transformer, tokenizer=tokenizer,
                       source=source_code, n_ctx=655360)


def load_baka_pythia():
    global source_code
    model_config = AutoConfig.from_pretrained(Path("~/models/pythia-160m-deduped").expanduser())
    model = AutoModelForCausalLM.from_config(model_config)
    window_size = 1024
    state_dict_path = f"weights/p2wiki103_p160_{model_numel(model)}_{window_size}.bin"
    print(f"loading {state_dict_path}")
    assert Path(state_dict_path).exists(), f"path not found {state_dict_path}"
    tokenizer = pythia_tokenizer()
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)
    model = model.to(device=device)
    source_code = Path("./bakanetproto2.py").read_text()
    return TestContext(model=model, main_model=model.gpt_neox, tokenizer=tokenizer,
                       source=source_code, n_ctx=1024)


def load_pythia_14m():
    return impl_load_transformers("pythia-14m")


def load_pythia_70m():
    return impl_load_transformers("pythia-70m-deduped")


def load_pythia_160m():
    return impl_load_transformers("pythia-160m-deduped")


def load_pythia_1b():
    return impl_load_transformers("pythia-1b-deduped")


def load_cerebras_111m():
    return impl_load_transformers("Cerebras-GPT-111M")


def load_cerebras_256m():
    return impl_load_transformers("Cerebras-GPT-256M")


def load_gpt2():
    return impl_load_transformers("gpt2", n_ctx=1024)


def load_open_llama3():
    return impl_load_transformers("open_llama_3b", dtype=torch.bfloat16)


def load_open_llama7():
    return impl_load_transformers("open_llama_7b", dtype=torch.bfloat16, n_ctx=1024)


def load_btlm():
    return impl_load_transformers("btlm-3b-8k-base", dtype=torch.bfloat16, trust_remote_code=True)


def load_mistral():
    return impl_load_transformers("Mistral-7B-v0.1", dtype=torch.bfloat16, n_ctx=1024)


def tokenize_text(samples, tokenizer):
    text = samples["text"]
    return {"input_ids": tokenizer(text).input_ids}


def get_dl(ctx: TestContext):
    tokenizer = ctx.tokenizer
    wiki_valid_ds_path = os.path.expanduser("~/Downloads/datasets/wikitext103/wiki.valid.jsonl")
    ds = datasets.load_dataset("json", data_files=[wiki_valid_ds_path])
    ds = ds.map(tokenize_text, remove_columns="text", fn_kwargs={"tokenizer": tokenizer}, batched=True)
    ds = ds["train"]
    dl = DataLoader(ds,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False))
    return dl


@torch.no_grad()
def run(ctx: TestContext):
    dl = get_dl(ctx)
    losses = []
    nans = 0
    bos_warned = False
    for b in tqdm(dl):
        if b['input_ids'][0, 0] == ctx.tokenizer.bos_token_id:
            if not bos_warned:
                bos_warned = True
                print("Warning, BOS injection nyi")

        n_seq = len(b['input_ids'][0])
        # TODO: test
        for i in range(0, n_seq, ctx.n_ctx//2):
            inputs = {k: v[:, i:i+ctx.n_ctx].to(device=device) for k, v in b.items()}
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
    con = sqlite3.connect("results/results.sqlite3")
    n_param = sum(p.numel() for p in model.parameters())
    main_model = ctx.main_model
    n_input_emb_params = sum(p.numel() for p in main_model.get_input_embeddings().parameters())
    benchmark = "wikipedia103"
    benchmark_split = "valid"
    with con as cursor:
        model_architecture_id = write_hashed_text(cursor, "ModelArchitecture", "architecture", str(model))
        source_id = None if not ctx.source else write_hashed_text(cursor, "Source", "source", ctx.source)

        cursor.execute("""INSERT INTO Result(id, model_type, 
                       n_params, n_input_emb_params, 
                       benchmark, benchmark_split,
                       nans, n_ctx, loss, ppl,
                       model_architecture_id, source_id) 
    VALUES (?, ?,  ?, ?,  ?,?, ?,?,?,?, ?,?)""", (
            record_id,
            model.config.model_type,
            n_param, n_input_emb_params,
            benchmark, benchmark_split,
            nans, n_ctx, loss, math.exp(loss),
            model_architecture_id, source_id))

        con.commit()
    return record_id


def main():
    print("Loading the model")
    # ctx = load_bakanet()
    # ctx = load_baka_pythia()
    # ctx = load_pythia_14m()
    # ctx = load_pythia_70m()
    # ctx = load_pythia_160m()
    ctx = load_pythia_1b()
    # ctx = load_cerebras_111m()
    # ctx = load_cerebras_256m()
    # ctx = load_gpt2()
    # ctx = load_open_llama3()
    # ctx = load_open_llama7()
    # ctx = load_btlm()
    # ctx = load_mistral()
    print("Loaded")

    # Sanity check
    assert batch_size == 1 or ctx.tokenizer.padding_side == "right", "LEFTPAD nyi"

    aloss, nans = run(ctx)
    print(f"AVGLOSS: {aloss:.5f}, PPL: {math.exp(aloss):.5f} NANS:{nans}, NCTX: {ctx.n_ctx}")
    record_id = record(ctx, aloss, nans)
    print(f"Recorded {record_id}")


if __name__ == "__main__":
    main()
