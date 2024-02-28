# BakaLLM: Sssnake, Ssssnake! (But faster)

* L8B8: Added  bunch of torch.compile. This saves memory a lot and allowed to run BS=8.
Unfortunately halving number of batches reduced overall performance to the level comparable with L12B5.

* L8B8D: Added dynamically built batches. 

Previously batches were "static" and "statically" split to minibatches.
E.g. if we have document D1=ABCDEFGHWX, D2=IJKLM, D3=NOPQR, with bach size = 2, ctx size = 4,
first batch would be

[ABCD, IJKL] then [EFGH, M___] (padding), then [WX] (end of D1)

Now system keeps track of finished batches and appends new documents:

[ABCD, IJKL] then [EFGH, M___] (padding), then [WX__, NOPQ] (end of D2 ended, so it was replaced by D3)

This massively improved performance. I got 3.77 after 1 epoch. Also definition of epoch was changed.
Previously 1 epoch ran over each document twice: one as usual, then with an offset n_ctx / 2.
Now only one pass is done per epoch. Originally I wanted to do both, but it worked so well, I now don't want to,

Also training became much "smoother". Previously loss jumped up and down as number of batches changed.
Now it improves steadily

## Training
(Scores are produced by bench.py, ie its validation split)

Mamba L8B8D: E1: 3.77266 (8 layers, with dynamic batches)


Mamba L8B8 E1:  4.26901 (8 layers w. mamba, 8 per batch for training)
           E2:  3.98698
           E3:  3.88880

Mamba L8 E1: B5: 4.23281 (8 layers, but each with mamba)
         E2: :   3.98490
         E3: :   3.87865

Old L12B5: E1: 4.26171875 (Mamba(Norm(x0+attn+mlp)) [each 2nd]
           E2: 3.99271
           E3: 3.88385 (new record, yay)

