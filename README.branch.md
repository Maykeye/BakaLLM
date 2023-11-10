# BakaLLM: Thin baka

Adding layers that have SelfAttn only to raise the number of tokens seen by XL

## Training

[-] Thin layer at start: severe degradation after 1st epoch
[-] Thin layer in the middle: comparable to long-reset 1ep despite having 9M more parameters

Thick layers make everything thin compared to selected layers

[+] Thick layer/naive at start: layer 0 in parallel calls another layer with attn and mlp. XL focuses on main layer attn; attn of other layer ignored
 aloss 4.53854 -- potentially good, but not 9M more good
[+] Thick layer/mlp at start: layer 0 has bigger MLP, attn not affected
 aloss 4.51641 -- potentially good, but not necessary 7M good. However so far it's the best post-e1 score
[+] Hourglass architecture:
 Right now each layer has the same MLP: 12 * 4 * dim_ff^2 [48 dimff]
 Instead we can use
 1 * 8 * dim_ff^2 [8]       [0]
 4 * 3 * dim_ff^2 [+12=20]  [4,5,6,7]
 7 * 4 * dim_ff^2 [+28=48]  [1,2,3, 8,9,10,11]
 For the same number of parameters
 aloss 4.50703 -- very good, as it's the same numbr of parameters. Worth checking all 3 epochs
 3ep;  4.0263  -- not impressive, the same as pause
[-] Thick layer/attn at start: layer 0 has bigger dim_attn, mlp not affected
 See below
[-] Thick layer/ at start: layer 0 has bigger dim_attn and MLP
 Considered non-worty pursuit: layer 2 needs to see KV cache of layer 1 to use in the XL attention.
 However if KV dim are different between layer 1 and 2, then a) I either need to cut dimensions in KV cache during cache save
 or to pass them through downcast which would require 2 FC and they will eat up a lot of parameters.
 Neither option looks promising
[ ] Non thick: increase dim_model across the board
[ ] Non thick: increase dim_attn across the board
[ ] Non thick: increase dim_ff across the board
  [-] for all:  increases number of parms
In case experiment will succeed, repeat, but give half parameters to last layer
