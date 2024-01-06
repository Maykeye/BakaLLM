# BakaFanOut: Reach for the stars!

Based on Llama* https://github.com/ggerganov/llama.cpp/discussions/4147

Base idea is to change hidden_size.

## Upscaler
pad-zero: valid AVGLOSS: E1: 4.4070 E3: 3.94401
  Idea that input will be shifted by input_norm, so essentially we are getting L[n+1] = cat(L[n], constant)
pad-rand: valid AVGLOSS: n/a, expected to be ~6.0, training was stop as mid E3 loss was around 6.0, which is 2.0 points worse. Not worse finishing
  Idea was to initialize with nn.Linear(): y = 🐱(x, nn.Linear(x, new_layer_dim-current_layer_dim))

head-rand
 Now idea take X, let's say it has form [a,a,b,b,c,c,d,d] (4 heads), then we reshape it into [[a a][b b][c c][d d]], so instead of padding whole X vector
 we pad lineary each head to get [[a a x] [b b y] [cc z] [d d w]]
 Loss still was around 2+ worse during training

head-zero: E1: 4.31615, E3: 3.90234 
similarr to pad zero, but this time zeros were appended after splitted dimension to n_heads parts, and then combining them  together.
Best result so far.

Fan-in: start at wide, narrow hidden size
Blech(2.0+ loss diff). Also broke RMT completely. Speaking of rmt.

head-zero-no-rmt: E1: 4.55234 E3: 3.98620 
Surprinsingly RMT works

const-width-mlp:
Best so far! E1: 4.32812, E3: 3.89323
Not by too much considering increased model size, but still the best. Moving to more epochs to see how it will play out

