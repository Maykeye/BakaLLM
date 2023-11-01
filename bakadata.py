import torch
import os
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Optional, Iterable


def tokenize_text(samples, tokenizer):
    text = samples["text"]
    return {"input_ids": tokenizer(text).input_ids}


def get_dl(tokenizer, batch_size, split="train", n_skip_batches=0):
    ds = load_dataset("iohadrubin/wikitext-103-raw-v1")
    assert isinstance(ds, DatasetDict)  # keep LSP happy
    ds = ds[split]
    assert "text" in ds.column_names
    ds = ds.map(
        tokenize_text,
        remove_columns=ds.column_names,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True)
    collator = DataCollatorForLanguageModeling(tokenizer, False)
    sampler = range(n_skip_batches * batch_size, len(ds))
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collator)
    return dl


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(os.path.expanduser("~/models/pythia-14m/"))
    tokenizer.pad_token_id = 0
    assert tokenizer.padding_side == "right"
    return tokenizer


@dataclass
class MiniBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    active_batches: Optional[torch.BoolTensor] = None
    last_active_batches: Optional[torch.BoolTensor] = None
    seqpos: int = 0
    seqtotal: int = 0

    @property
    def n_batch(self):
        return self.input_ids.shape[0]

    @property
    def progress(self):
        if not self.seqtotal:
            return 0
        return self.seqpos/self.seqtotal

    def pass_batch_from_past_to_present(self, tensor: torch.Tensor):
        """ Iterator can cut off batches that no longer have useful information. 
        However we may need to pass batches from the past leaving only batches that still exist in presence. 
        This function does so: if current minibatch removed batches wrt to the last one, they are also removed from the tensor"""
        # check if we cut anything at all
        if self.active_batches is None or self.last_active_batches is None:
            return tensor

        pick_from_past = (self.active_batches == self.last_active_batches)[self.last_active_batches]
        return tensor[pick_from_past]


def batch_iterator(
        batch,
        n_ctx,
        n_stride,
        n_skip_first=0,
        device="cuda",
        tokenizer: Optional[AutoTokenizer] = None,
        cut_empty=True
) -> Iterable[MiniBatch]:
    """Split the bigger batch into batches small enough

    Args:
        batch (torch.Tensor): the original batch(e.g. wikitext articles)
        n_ctx (_type_): context size of the minibatch
        n_stride (_type_): stride (step)
        n_skip_first (int, optional): Skip first N tokens from the global batch. Defaults to 0.
        device (str, optional): Pass each mini-batch to this device. Defaults to "cuda".
        tokenizer (Optional[AutoTokenizer], optional): Tokenizer for decided which empty batches to discard. Defaults to None.
        cut_empty (bool, optional): Flag that sets if empty batches(batches that contain padding only) should be discarded and not returned to the caller. Defaults to True.

    Yields:
        Iterator[MiniBatch]: Mini-batch
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    if n_skip_first:
        input_ids = input_ids[:, n_skip_first:]
        attention_mask = attention_mask[:, n_skip_first:]
        labels = labels[:, n_skip_first:]
    assert tokenizer.padding_side == 'right'
    last_mask = None
    mb_select_mask = None
    seq_total = input_ids.shape[-1]
    for i in range(0, seq_total, n_stride):
        mb_inputs = input_ids[:, i:i+n_ctx].to(device=device)
        if mb_inputs.shape[1] < 3:
            break
        mb_labels = labels[:, i:i+n_ctx].to(device=device)
        mb_attn_mask = attention_mask[:, i:i+n_ctx].to(device=device)
        # do not run empty batches
        if cut_empty and tokenizer.pad_token_id is not None:
            last_mask = mb_select_mask
            # Check for <pad> at position 1 as  pos 0 might contain <bos> which might be equal to <pad>
            mb_select_mask = mb_attn_mask[:, 1] != tokenizer.pad_token_id
            mb_inputs = mb_inputs[mb_select_mask]
            mb_labels = mb_labels[mb_select_mask]
            mb_attn_mask = mb_attn_mask[mb_select_mask]
        # TODO: BOS injection
        # TODO: <pause> injection

        yield MiniBatch(mb_inputs, mb_attn_mask, mb_labels, mb_select_mask, last_mask, seqpos=i, seqtotal=seq_total)
