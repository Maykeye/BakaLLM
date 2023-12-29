# BakaFanOut: Reach for the stars!

Based on Llama* https://github.com/ggerganov/llama.cpp/discussions/4147

Base idea is to change hidden_size.

## Training
Upscaler-pad-zero-fix: valid AVGLOSS: E1: 4.4070 E3: 3.94401
  Idea that input will be shifted by input_norm, so essentially we are getting L[n+1] = cat(L[n], constant)


