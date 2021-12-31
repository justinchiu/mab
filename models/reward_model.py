import pomdp_py

class RewardModel(pomdp_py.RewardModel):
    def _reward_func(self, state, action):
        if isinstance(action, Ask):
            return -action.id.sum()
        elif isinstance(action, Select):
            return state[action.id].prob

    def sample(self, state, action, next_state):
        # deterministic?
        return self._reward_func(state, action)
