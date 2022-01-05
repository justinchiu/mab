import pomdp_py

from models.policy_model import PolicyModel
from models.transition_model import ProductTransitionModel
from models.observation_model import ObservationModel
from models.reward_model import RewardModel

from agent.belief import initialize_belief

class RsAgent(pomdp_py.Agent):
    def __init__(
        self,
        num_dots,
        belief_rep,
        prior,
        num_particles=100,
        num_bins = 5,
    ):
        self.id = 0
        init_belief = initialize_belief(
            num_dots,
            prior,
            belief_rep,
            num_particles,
            num_bins,
        )
        policy_model = PolicyModel(num_dots)
        transition_model = ProductTransitionModel(num_dots)
        observation_model = ObservationModel(num_dots)
        reward_model = RewardModel()
        super().__init__(
            init_belief,
            policy_model,
            transition_model = transition_model,
            observation_model = observation_model,
            reward_model = reward_model,
        )

    def clear_history(self):
        self._history = None
