import random
import numpy as np
import pomdp_py

from mab.beta_bernoulli.domain.action import Action, Ask, Select, Pass
from mab.beta_bernoulli.domain.state import Go, Stop

def generate_all_boolean_vectors(size):
    a = np.arange(2 ** size, dtype=np.uint8)[:,None]
    return np.unpackbits(a, axis=1)[1:,-size:]

def convert_numpy(history):
    asks = [a.val for (a,o) in history if isinstance(a, Ask)]
    obs = [tuple(o.obs.values()) for (a,o) in history if isinstance(a, Ask)]
    return np.vstack(asks) if asks else asks, np.array(obs) if obs else obs

def get_belief_order(state, num_dots):
    probs = [
        obj.attributes["prob"][1]
        for dot, obj in state.object_states.items()
        if dot < num_dots
    ]
    probs = np.array(probs)
    return (-probs).argsort()

def one_hot_bool(size, idx):
    zero = np.zeros(size, dtype=np.bool)
    zero[idx] = True
    return zero

def get_fresh_dot(belief_order, asks):
    for dot in belief_order:
        if not asks[dot]:
            return dot

def get_selected_dot(asks, obs):
    idx, dot = (obs == 1).nonzero()
    # tells us the ask
    return dot[0]

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
        self.ASK = [Ask(np.array([
            x == id for x in range(num_dots)
        ], dtype=bool)) for id in range(num_dots)]
        # ASK = all binary vectors of size num_dots
        #all_vecs = generate_all_boolean_vectors(num_dots)
        #self.ASK = [Ask(x) for x in all_vecs]
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

    def rollout(self, state, history):
        num_dots = self.num_dots
        countdown = state.object_states[num_dots + 1].t
        belief_order = get_belief_order(state, num_dots)
        if len(history) == 0:
            return Ask(one_hot_bool(num_dots, belief_order[0]))

        last_action, last_obs = history[-1]
        if isinstance(last_action, Select) or isinstance(last_action, Pass):
            return Pass()

        asks, obs = convert_numpy(history)
        if (obs == 1).any():
            # select
            idx = get_selected_dot(asks, obs)
            return Select(idx)
        else:
            idx = get_fresh_dot(belief_order, asks.any(0))
            if idx is None:
                # no arms yielded good answers
                return Select(0)
            return Ask(one_hot_bool(num_dots, idx))

if __name__ == "__main__":
    print("Testing policy model")
