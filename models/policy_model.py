import pomdp_py

from domain.action import Action, Ask, Select, Pass
from domain.state import Go, Stop

class PolicyModel(pomdp_py.RandomRollout):
    """A simple policy model with uniform prior over a
       small, finite action space"""
    def __init__(self, num_dots):
        super().__init__()
        self.num_dots = num_dots
        self.SELECT = [Select(id) for id in range(num_dots)]
        self.ASK = [Ask(id) for id in range(num_dots)]
        self.ACTIONS = self.SELECT + self.ASK

    def sample(self, state, **kwargs):
        return self.get_all_actions().random()

    def get_all_actions(self, state=None, history=None):
        return self.ACTIONS
        """
        robot_state = state.object_states[0]
        if isinstance(robot_state, Go):
            return self.ACTIONS
        elif isinstance(robot_state, Stop):
            return [Pass()]
        """

if __name__ == "__main__":
    print("Testing policy model")
