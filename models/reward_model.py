import pomdp_py

from domain.action import Ask, Select
from domain.state import Go, Stop

class RewardModel(pomdp_py.RewardModel):
    def __init__(self, num_dots):
        super().__init__()
        self.robot_id = num_dots
        self.countdown_id = num_dots + 1

    def _reward_func(self, state, action):
        robot_state = state.object_states[self.robot_id]
        countdown_state = state.object_states[self.countdown_id]
        if isinstance(robot_state, Stop):
            # state transitions to Stop after selection has been made
            # always give 0 reward after selection has been made
            return 0
        elif countdown_state.t < 0:
            # penalize if the game finishes without selection,
            # ie robot_state != Stop
            return -100
        elif isinstance(action, Ask):
            #return -action.val.sum()
            #return -.1
            return -1
        elif isinstance(action, Select):
            #return state[action.val].prob
            # increment action.val by 1, since 0th state is the robot
            #return state.object_states[action.val]["prob"]
            if state.object_states[action.val]["prob"] > 0.5:
                return 10
            else:
                return -100


    def sample(self, state, action, next_state):
        # deterministic?
        return self._reward_func(state, action)
