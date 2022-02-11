import numpy as np
import pomdp_py

from domain.action import Action, Ask, Select, Pass
from domain.state import Go, Stop

def generate_all_boolean_vectors(size):
    a = np.arange(2 ** size, dtype=np.uint8)[:,None]
    return np.unpackbits(a, axis=1)[1:,-size:]

class PolicyModel(pomdp_py.RandomRollout):
    """A simple policy model with uniform prior over a
       small, finite action space"""
    def __init__(self, num_dots, max_turns = 5):
        super().__init__()
        self.num_dots = num_dots
        self.max_turns = max_turns

        self.SELECT = [Select(id) for id in range(num_dots)]
        #self.ASK = [Ask([id]) for id in range(num_dots)]
        #self.ASK = [Ask(id) for id in range(num_dots)]
        #self.ASK = [Ask(np.array([
            #x == id for x in range(num_dots)
        #], dtype=bool)) for id in range(num_dots)]
        # ASK = all binary vectors of size num_dots
        all_vecs = generate_all_boolean_vectors(num_dots)
        self.ASK = [Ask(x) for x in all_vecs]
        self.ACTIONS = self.SELECT + self.ASK
        self.PASS = [Pass()]

    def sample(self, state, **kwargs):
        return self.get_all_actions().random()

    def get_all_actions(self, state=None, history=None):
        # for debugging, just allow PASS always
        return self.ACTIONS + self.PASS

        if state.object_states[self.num_dots+1].t == self.max_turns:
            # first turn, allow Pass
            return self.ACTIONS + self.PASS
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
