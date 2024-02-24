# BakaLLM: Sssnake, Ssssnake!



## Training

MiniE1: B9 4.53802061080933 MiniMamba: layers got halved to half the training speed(it didn't halved)
MiniE3: B8 4.04557275772095 (Compared to RMT B5: 3.99557280540466)
MineE1: B8 4.27630186080933 (Mamba(norm(x0+attn+mlp)) + x0)
Seq E1; B5: 4.5632 (Mamba(MLP(Attn))
Seq E1; B5: 4.47630 (MLP(attn(mamba)))

Proper E1 : B5: 4.26171875 (Mamba(Norm(x0+attn+mlp)) [each 2nd]
 Proper E2: B5: 3.99271
 Proper E3: B5: 3.88385 (new record, yay)
Proper E1 : B5: 4.36745 [each 4th]
       E2 :   : 4.05729
       E3 :   : 3.94271

Mamba L8 E1: B5: 4.23281 (8 layers, but each with mamba)
         E2: :   3.98490
         E3: :   3.87865
