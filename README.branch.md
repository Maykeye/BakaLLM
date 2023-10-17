# BakaLLM: XL Repos Edition

Experiments with XL

## Architecture 

The base idea from https://arxiv.org/abs/1901.02860 was implemented.
For positional embedding rotary embeddings were used. 
Keys always start at position 0.

Experiment 1: query are offsetted to position `n_past`.
Result: Epoch 1: 4.926 with Q repositioning, 4.793 for Q+K repositioning (original 00_xl brancxh). That's a big oof

Experiment 2: nothing is offsetted, both (q_now) and (k_past 🐱 k_now) have offset offset of 0, which means query think its keys are the past
Result: TBD

This emulates two things
* Relative Positions
* StreamingLLM(https://arxiv.org/abs/2309.17453) in KV cache does the same


## Training
TODO: training schedule

TODO: graph

