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
[ ] Thick layer/attn at start: layer 0 has bigger dim_attn, mlp not affected
 ^ current focus to be
[ ] Thick layer/ at start: layer 0 has bigger dim_attn and MLP
[ ] Non thick: increase dim_model
[ ] Non thick: increase dim_ff

In case experiment will succeed, repeat, but give half parameters to last layer
