# BakaLLM: XL Repos Edition

Experiments with XL

## Architecture 

The base idea from https://arxiv.org/abs/1901.02860 was implemented.
For positional embedding rotary embeddings were used. 
Keys always start at position 0.

Experiment 1: query are offsetted to position `n_past`.
Result: Epoch 1: 4.926 with Q repositioning, 4.793 for Q+K repositioning (original 00_xl brancxh). That's a big oof

This emulates two things
* Relative Positions
* StreamingLLM(https://arxiv.org/abs/2309.17453) in KV cache does the same

Experiment 2: nothing is offsetted, both (q_now) and (k_past 🐱 k_now) have offset offset of 0, which means query think its keys are the past
Result: DNF, significantly worse after 15K steps, not worth finishing (5.7 vs 5.4 repos)

Experiment 3: Theta=2k
Intuition: bfloat16 is not precise enough to rotate aroun 10K times. 2K is more than enough:
512 is pure size, 512 is history size, 1K left for RMT, etc
DNF: same as exp 1

Experiment 4: Long/no-reset
Training schedule is changed.
Minibatches no longer overlap(to do that, minibatch is run for the second time with ctx_size//2 offset),
But old states are passed within minibatch
(Exp 4.5 DNF: Due to implementation error, first run the same batch was trained on twice per step(no offset in ctx_size and not so surprisigly it went down much faster than any previous one, I may need to reinvestiage it later)

DNF: OoM after 100 minutes due to memory spike

Experiment 5: Long/no-reset
Good: 4.14817714691162 after 3 epochs

Experiment 6: Long/flip-flip/5k

Consider sequence splitted to windows 
aaaa bbbb cccc dddd
Rotation is done like this:

window: aaaa bbbb cccc dddd
offset: 0000 5000 0000 5000

aaa is rotated to be at 0th offset
bbb is rotated to be at 5000th offset, historical aaa is still at 0.
ccc is rotated to be at 0th offset. historical bbb is still at 5000
ddd is rotated to 5000 offset, past ccc is still rotated at 0th offset.

Hypothesis of why reset doesn't work:
model learns that token is located at N-th position. If the position changed, model experiences shock.
If offset is ever-increasing, model learns every offset eventually.

Here comes flip-flop:
model will learn that window at 0000 for the past looks at position 5000. 
at the same time, model will learn that sequence at position 5000 looks at position 0000, completing the circle.

It can learn to some trobules in memory retrival phase, but that's a trouble for later and can be solved by using solidifier layer/etc

## Training
TODO: training schedule

TODO: graph

