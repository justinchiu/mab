import pomdp_py

from mab.beta_bernoulli.domain.action import Ask, Select, Pass
from mab.beta_bernoulli.domain.state import Go, Stop

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
        elif countdown_state.t <= 0:
            # penalize if the game finishes without selection,
            # ie robot_state != Stop
            return -100
        elif isinstance(action, Ask) or isinstance(action, Pass):
            #return -action.val.sum()
            #return -.1
            return -1
        elif isinstance(action, Select):
            win = 10
            fail = -100
            prob = state.object_states[action.val]["prob"]
            # prob gives the probability a reference resolves to n diff dots
            prob_win = prob[1]
            return prob_win * win + (1 - prob_win) * fail
        else:
            raise ValueError(f"Invalid action: {action}")


    def sample(self, state, action, next_state):
        # deterministic?
        return self._reward_func(state, action)

if __name__ == "__main__":
    print("Testing reward model")
