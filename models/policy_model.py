import pomdp_py

from domain.action import Action, Ask, Select

class PolicyModel(pomdp_py.RandomRollout):
    """A simple policy model with uniform prior over a
       small, finite action space"""
    def __init__(self, num_dots):
        super().__init__()
        self.num_dots = num_dots
        self.ACTIONS = (
            [Ask(id) for id in range(num_dots)]
            + [Select(id) for id in range(num_dots)]
        )

    def sample(self, state, **kwargs):
        return self.get_all_actions().random()

    def get_all_actions(self, **kwargs):
        return self.ACTIONS
