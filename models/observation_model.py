
import numpy as np
import pomdp_py

from domain.observation import ArmObservation, ProductObservation

class ObservationModel(pomdp_py.OOObservationModel):
    def __init__(self, num_dots):
        #observation_models = [ArmObservationModel(id) for id in range(num_dots)]
        observation_models = {id: ArmObservationModel(id) for id in range(1, num_dots+1)}
        # no arm observation model
        super().__init__(observation_models)

    def sample(self, next_state, action, argmax=False, **kwargs):
        factored_observations = super().sample(next_state, action, argmax=argmax)
        # do a max depending on bernoulli or structured feedback
        return ProductObservation.merge(factored_observations, next_state)

class ArmObservationModel(pomdp_py.ObservationModel):
    """ Frequentist bandit
    """
    def __init__(self, id):
        self.id = id

    def probability(self, observation, next_state, action, **kwargs):
        # p(observation | next_state, action)
        # for the beta bernoulli setting
        #alpha = next_state.alpha
        #t = next_state.t
        #prob = alpha / t
        #return prob if observation else 1 - prob
        prob = next_state.object_states[self.id]["prob"]
        return prob if observation else 1 - prob

    def sample(self, next_state, action, **kwargs):
        prob = next_state.object_states[self.id]["prob"]
        # probably needs to condition on action, i.e.
        y = np.random.binomial(1, prob)
        # return y if action[self.id], y ~ Bern(next_state.prob)
        return ArmObservation(self.id, y)

    def argmax(self, next_state, action, **kwargs):
        raise NotImplementedError

    #def get_all_observations(self):
        #raise NotImplementedError

