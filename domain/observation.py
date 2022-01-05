import pomdp_py

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

    def __str__(self):
        return f"ArmObs {self.objid} ({self.feedback})"

class ProductObservation(pomdp_py.OOObservation):
    """Observation for MAB that can be factored by arms;
    thus this is an OOObservation."""
    def __init__(self, obs):
        """
        obs (list): list of boolean arm observations (not ObjectObservation!).
        actually, not a list.
        obs (dict): list of boolean arm observations (not ObjectObservation!).
        """
        self._hashcode = hash(frozenset(obs))
        self.obs = obs

    def for_obj(self, id):
        if id in self.obs:
            return ArmObservation(id, self.obs[id])
        else:
            return ArmObservation(id, ObjectObservation.NULL)

    def __hash__(self):
        return self._hashcode
    
    def __eq__(self, other):
        if not isinstance(other, ProductObservation):
            return False
        else:
            return self.obs == other.obs

    def __str__(self):
        return "ProductObservation(%s)" % str(self.obs)

    def __repr__(self):
        return str(self)

    def factor(self, next_state, *params, **kwargs):
        """Factor this OO-observation by objects"""
        #return [ArmObservation(idx, fb) for (idx, fb) in enumerate(self.obs)]
        return {
            id: ArmObservation(idx, fb)
            for (idx, fb) in self.obs.items()
        }
    
    @classmethod
    def merge(cls, obs, next_state, *params, **kwargs):
        """Merge `object_observations` into a single OOObservation object"""
        return ProductObservation({
            id: arm_obs.feedback for (id, arm_obs) in obs.items()
        })

