# Multi-Armed Bandit POMDP
# pomdp-py implementation
# based off of pomdp_py/pomdp_problems/multi_object_search

import pomdp_py
import numpy as np

# Domain

NUM_DOTS = 3

# STATES
GO = 0
STOP = 1
 
class ArmState(pomdp_py.ObjectState):
    def __init__(self, id, prob, shape, color, xy):
        super().__init__()
        self.id = id
        self.prob = prob
        self.shape = shape
        self.color = color
        self.xy = xy

    def __str__(self):
        return f"Dot {self.xy} {self.shape} {self.color}"

class AgentState(pomdp_py.ObjectState):
    # not sure this is necessary
    def __init__(self, id):
        super().__init__()
        self.id = id
        self.state = GO

    def __str__(self):
        return f"Agent {self.id}"

# ACTIONS

class Action(pomdp_py.Action):
    """Bandit action; Simple named action."""
    def __init__(self, name):
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, Action):
            return self.name == other.name
        elif type(other) == str:
            return self.name == other
    def __str__(self):
        return self.name
    def __repr__(self):
        return "Action(%s)" % self.name

class Ask(Action):
    def __init__(self, dotid):
        super().__init__(dotid)

# OBSERVATIONS

class ArmObservation(pomdp_py.Observation):
    """Binary feedback for a single arm"""
    def __init__(self, objid, feedback):
        self.objid = objid
        self.feedback = feedback

    def __hash__(self):
        return hash((self.objid, self.feedback))
    def __eq__(self, other):
        if not isinstance(other, ObjectObservation):
            return False
        else:
            return (
                self.objid == other.objid
                and self.feedback == other.feedback
            )

class OoObservation(pomdp_py.OOObservation):
    """Observation for MAB that can be factored by arms;
    thus this is an OOObservation."""
    def __init__(self, obs):
        """
        obs (list): list of boolean arm observations (not ObjectObservation!).
        """
        self._hashcode = hash(frozenset(obs))
        self.obs = obs

    def __hash__(self):
        return self._hashcode
    
    def __eq__(self, other):
        if not isinstance(other, MosOOObservation):
            return False
        else:
            return self.obs == other.obs

    def __str__(self):
        return "OoObservation(%s)" % str(self.obs)

    def __repr__(self):
        return str(self)

    def factor(self, next_state, *params, **kwargs):
        """Factor this OO-observation by objects"""
        return [ArmObservation(idx, fb) for (idx, fb) in enumerate self.obs]
    
    @classmethod
    def merge(cls, obs, next_state, *params, **kwargs):
        """Merge `object_observations` into a single OOObservation object"""
        return OoObservation([o.feedback for o in obs])

# Model
 
# Observation Model
class ObservationModel(pomdp_py.OOObservationModel):
    def __init__(self, num_dots):
        observation_models = [ArmObservationModel(id) for id in range(num_dots)]
        super().__init__(observation_models)

    def sample(self, next_state, action, argmax=False, **kwargs):
        factored_observations = super().sample(next_state, action, argmax=argmax)
        # do a max depending on bernoulli or structured feedback
        return OoObservation.merge(factored_observations, next_state)

class ArmObservationModel(pomdp_py.ObservationModel):
    """ Frequentist bandit
    """
    def __init__(self, id):
        self._id = id

    def probability(self, observation, next_state, action, **kwargs):
        # p(observation | next_state, action)
        # for the beta bernoulli setting
        #alpha = next_state.alpha
        #t = next_state.t
        #prob = alpha / t
        #return prob if observation else 1 - prob
        return next_state.prob if observation else 1 - next_state.prob

    def sample(self, next_state, action, **kwargs):
        import pdb; pdb.set_trace()
        # probably needs to condition on action, i.e.
        y = np.random.binomial(1, next_state.prob)
        # return y if action[self._id], y ~ Bern(next_state.prob)
        return ArmObservation(self._id, y)

    def argmax(self, next_state, action, **kwargs):
        raise NotImplementedError

    #def get_all_observations(self):
        #raise NotImplementedError

# Transition Model
class TransitionModel(pomdp_py.OOTransitionModel):
    def __init__(self):
        pass

    def sample(self):
        pass

class ArmTransitionModel(pomdp_py.TransitionModel):
    """ Static
    """
    def __init__(self):
        pass

class AgentTransitionModel(pomdp_py.TransitionModel):
    def __init__(self):
        pass

# Reward Model
class RewardModel(pomdp_py.RewardModel):
    def __init__(self):
        pass

arms = [
    (0, "large", "black", (0,0)),
    (1, "large", "white", (0,1)),
    (2, "small", "grey",  (0,2)),
]

print(ArmState.attributes)
