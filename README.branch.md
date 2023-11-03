# BakaLLM: Split Layer Norm

Previously layer norm was one per whole layer, as in stable alpha
Now it's one per each layer part, as in gpt neox.

## Training
Not much changed tbh. One run was way worse, one was slightly better.
Inclusion in mainline is not yet decided
