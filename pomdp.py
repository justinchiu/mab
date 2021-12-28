# Multi-Armed Bandit POMDP
# pomdp-py implementation
# based off of pomdp_py/pomdp_problems/multi_object_search

import pomdp_py

# STATES
GO = 0
STOP = 1
 
class ArmState(pomdp_py.ObjectState):
    def __init__(self, id, shape, color, xy):
        super().__init__()
        self.id = id
        self.shape = shape
        self.color = color
        self.xy = xy

    def __str__(self):
        return f"Dot {self.xy} {self.shape} {self.color}"

class AgentState(pomdp_py.ObjectState):
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
        feedback (list): list of boolean feedbacks (not ObjectObservation!).
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
        #return {objid: ObjectObservation(objid, self.objattrs[objid])
                #for objid in next_state.object_states
                #if objid != next_state.robot_id}
    
    @classmethod
    def merge(cls, obs, next_state, *params, **kwargs):
        """Merge `object_observations` into a single OOObservation object"""
        return OoObservation([o.feedback for o in obs])

# REWARDS


arms = [
    (0, "large", "black", (0,0)),
    (1, "large", "white", (0,1)),
    (2, "small", "grey",  (0,2)),
]

print(ArmState.attributes)
