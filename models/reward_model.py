import pomdp_py

from domain.action import Ask, Select

class RewardModel(pomdp_py.RewardModel):
    def _reward_func(self, state, action):
        if isinstance(action, Ask):
            #return -action.val.sum()
            return -1
        elif isinstance(action, Select):
            return state[action.val].prob

    def sample(self, state, action, next_state):
        # deterministic?
        return self._reward_func(state, action)
