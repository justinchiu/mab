# POMDP approaches to the multi-arm bandit problem

Comparison of exact dynamic programming (value iteration)
to UCB / Thompson sampling for the best arm identification variant of the MAB problem.
Rather than receiving reward at each pull,
reward is only recieved at a terminal state and each action has some associated cost.

Current status:
* PO-UCT works relatively well for small scale problems, but is extremely sample inefficient.
* POMCP seems to work better than PO-UCT, but is very slow.

TODO:
* Exact inference
* Permutation invariance
* Feature-based POMDP
