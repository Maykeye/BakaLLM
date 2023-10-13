# BakaLLM: Making LLM from scratch
This is a main branch. It contains nothing.
Experiments are done in subbranches like `001-pristine`

Overall the goal of this experiment is to make a model that uses a lot memory techniques(RMT, XL, etc).
Previous attempt can be seen in the branch 000_OLD.

Note. While the source is available under APACHE2.0, running it will be not straightforward as it's a pet
project. Inference script will not be done for a while. Benchmark script requires specific $HOME directory layout with models placed in $HOME/models.

Branches:

* 001_pristine

Base implementation. It's the 'fastest' (or at least simplest) and serves as current starting point

* 00x_actfn_elephant

Experimental branch(hence 00x prefix) using elephant actfn