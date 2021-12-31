
import pomdp_py

class ObservationModel(pomdp_py.OOObservationModel):
    def __init__(self, num_dots):
        observation_models = [ArmObservationModel(id) for id in range(num_dots)]
        super().__init__(observation_models)

    def sample(self, next_state, action, argmax=False, **kwargs):
        factored_observations = super().sample(next_state, action, argmax=argmax)
        # do a max depending on bernoulli or structured feedback
        return ProductObservation.merge(factored_observations, next_state)

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

