# BakaLLM: Sssnake, Ssssnake! (But faster)

L8B8: Added  bunch of torch.compile. This saves memory a lot and allowed to run BS=8.
Unfortunately halving number of batches reduced overall performance to the level comparable with L12B5.

## Training
Mamba L8B8 E1:  4.26901 (8 layers w. mamba, 8 per batch for training)
           E2:  3.98698
           E3:  3.88880

Mamba L8 E1: B5: 4.23281 (8 layers, but each with mamba)
         E2: :   3.98490
         E3: :   3.87865

Old L12B5: E1: 4.26171875 (Mamba(Norm(x0+attn+mlp)) [each 2nd]
           E2: 3.99271
           E3: 3.88385 (new record, yay)

