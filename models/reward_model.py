import pomdp_py

from domain.action import Ask, Select
from domain.state import Go, Stop

class RewardModel(pomdp_py.RewardModel):
    def _reward_func(self, state, action):
        robot_state = state.object_states[0]
        if isinstance(robot_state, Stop):
            return 0
        elif isinstance(action, Ask):
            #return -action.val.sum()
            #return -.1
            return -1
        elif isinstance(action, Select):
            #return state[action.val].prob
            # increment action.val by 1, since 0th state is the robot
            #return state.object_states[action.val+1]["prob"]
            if state.object_states[action.val+1]["prob"] > 0.5:
                return 10
            else:
                return -100


    def sample(self, state, action, next_state):
        # deterministic?
        return self._reward_func(state, action)
