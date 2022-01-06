
import numpy as np
import pomdp_py

from domain.action import Ask, Select
from domain.observation import ArmObservation, ProductObservation

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
        # for the beta bernoulli setting
        #alpha = next_state.alpha
        #t = next_state.t
        #prob = alpha / t
        #return prob if observation else 1 - prob
        #prob = next_state.object_states[self.id]["prob"]
        if isinstance(action, Ask):
            prob = next_state["prob"]
            return prob if observation.feedback else 1 - prob
        elif isinstance(action, Select):
            return 0.01 if observation.feedback else 0.99
        else:
            return ValueError(f"Invalid action: {action}")

    def sample(self, next_state, action, **kwargs):
        if isinstance(action, Ask):
            prob = next_state.object_states[self.id]["prob"]
            # TODO: handle vector-valued action
            y = np.random.binomial(1, prob) if action.val == self.id else 0
            # return y if action[self.id], y ~ Bern(next_state.prob)
            return ArmObservation(self.id, y)
        elif isinstance(action, Select):
            return ArmObservation(self.id, 0)
        else:
            raise ValueError(f"Invalid action: {action}")

    def argmax(self, next_state, action, **kwargs):
        raise NotImplementedError

    #def get_all_observations(self):
        #raise NotImplementedError

if __name__ == "__main__":
    print("Testing observation models")
