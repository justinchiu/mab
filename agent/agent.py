import pomdp_py

class RsAgent(pomdp_py.Agent):
    def __init__(
        self,
        id,
        belief_rep,
        prior,
        num_particles=100,
    ):
        init_belief = None
        policy_model = PolicyModel()
        transition_model = AgentTransitionModel()
        observation_model = ObservationModel()
        reward_model = RewardModel()
        super().__init__(
            init_belief,
            policy_model,
            transition_model = transition_model,
            observation_model = observation_model,
            reward_model = reward_model,
        )

