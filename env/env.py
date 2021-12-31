import pomdp_py

class RsEnvironment(pomdp_py.Environment):
    def __init__(self, num_dots, init_state):
        transition_model = TransitionModel()
        reward_model = RewardModel()
        super().__init__(init_state, transition_model, reward_model)
