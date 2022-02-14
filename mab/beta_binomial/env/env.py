import numpy as np
import pomdp_py

from mab.beta_bernoulli.models.transition_model import ProductTransitionModel
from mab.beta_binomial.models.reward_model import RewardModel

class RsEnvironment(pomdp_py.Environment):
    def __init__(self, num_dots, dot_vector, init_state):
        self.num_dots = num_dots
        self.dot_vector = dot_vector

        transition_model = ProductTransitionModel(num_dots)
        reward_model = RewardModel(num_dots)
        super().__init__(init_state, transition_model, reward_model)
