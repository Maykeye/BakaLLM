# BakaLLM: Let's pause for a second

Added pauses inspired by https://arxiv.org/abs/2310.02226 
Model adds psuedo tokens randomly(they are not part of vocablurary), with
exception of adding token at position 0, to get that sweet BOS imitation.
This also means that ppl is no longer trully deterministic as pause
are random. 

However after measuring loss on valid split two times, loss was 4.02xx
both times, so I'll fixed it later(maybe never)

## Training
![Training graph](./train_pause.png)
For the first time training fell below 4.00
