# BakaLLM: XL Repos Edition

Experiments with XL

## Architecture 

The base idea from https://arxiv.org/abs/1901.02860 was implemented.
For positional embedding rotary embeddings were used. 
Keys always start at position 0.
Query are offsetted to position `n_past`.

This emulates two things
* Relative Positions
* StreamingLLM(https://arxiv.org/abs/2309.17453) in KV cache does the same


## Training
TODO: training schedule

TODO: graph

