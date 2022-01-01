# Multi-Armed Bandit POMDP
# pomdp-py implementation
# based off of pomdp_py/pomdp_problems/multi_object_search
 
import numpy as np

import pomdp_py # type: ignore

from problem import RankingAndSelectionProblem, solve

NUM_DOTS = 3

arms = [
    (0, "large", "black", (0,0)),
    (1, "large", "white", (0,1)),
    (2, "small", "grey",  (0,2)),
]


