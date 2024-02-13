# BakaLLM: Mini snake
First number of layers got halfed to half the training speed


## Training

MiniE1: B9 4.53802061080933
MiniE3: B8 4.04557275772095 (Compared to RMT B5: 3.99557280540466)
MineE1: B8 4.27630186080933 (Mamba(norm(x0+attn+mlp)) + x0)
~~ProperE1: B5: 4.10598945617676 (Mamba each 2nd layer): it seems I grabbed half of E2~~
Proper E1 : B5: 4.26171875 (Mamba(Norm(x0+attn+mlp))
 Proper E2: B5: 3.99271
 Proper E2: B5: 3.88385 (new record, yay)
