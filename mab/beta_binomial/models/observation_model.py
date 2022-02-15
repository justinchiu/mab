
import random

import numpy as np
import pomdp_py

from mab.beta_bernoulli.domain.action import Ask, Select, Pass
from mab.beta_bernoulli.domain.observation import ArmObservation, ProductObservation

class ObservationModel(pomdp_py.OOObservationModel):
    def __init__(self, num_dots):
        #observation_models = [ArmObservationModel(id) for id in range(num_dots)]
        observation_models = {
            id: ArmObservationModel(id) for id in range(num_dots)
        }
        # no agent observation model
        super().__init__(observation_models)

    def sample(self, next_state, action, argmax=False, **kwargs):
        factored_observations = super().sample(next_state, action, argmax=argmax)
        # do a max depending on bernoulli or structured feedback
        return ProductObservation.merge(factored_observations, next_state)

class ArmObservationModel(pomdp_py.ObservationModel):
    """ Frequentist bandit
    """
    def __init__(self, id):
        # arm id
        # in belief state and world state, objid = self.id
        self.id = id

    def probability(self, observation, next_state, action, **kwargs):
        # p(observation | next_state, action)
        # observation in {1, ..., num_dots}?
        if isinstance(action, Ask):
            prob = next_state["prob"]
            return prob[observation.feedback]
        elif isinstance(action, Select):
            # arbitrary
            return 0.01 if observation.feedback != 0 else 0.99
        elif isinstance(action, Pass):
            # arbitrary
            return 0.01 if observation.feedback != 0 else 0.99
        else:
            return ValueError(f"Invalid action: {action}")

    def sample(self, next_state, action, **kwargs):
        if isinstance(action, Ask):
            if not action.val[self.id]:
                return ArmObservation(self.id, 0)
            prob = next_state.object_states[self.id]["prob"]
            num_dots = prob.shape[0]
            y = np.random.choice(a=num_dots, p=prob)
            return ArmObservation(self.id, y)
        elif isinstance(action, Select):
            return ArmObservation(self.id, 0)
        elif isinstance(action, Pass):
            return ArmObservation(self.id, 0)
        else:
            raise ValueError(f"Invalid action: {action}")

    def argmax(self, next_state, action, **kwargs):
        raise NotImplementedError

    #def get_all_observations(self):
        #raise NotImplementedError

if __name__ == "__main__":
    print("Testing observation models")
