import pomdp_py

class TransitionModel(pomdp_py.OOTransitionModel):
    def __init__(self, num_dots, epsilon=1e-9):
        self._epsilon = epsilon
        transition_models = [
            ArmTransitionModel(id) for id in range(num_dots)
        ]
        super().__init__(transition_models)


    def sample(self, state, action, **kwargs):
        prod_state = pomdp_py.OOTransitionModel.sample(self, state, action, **kwargs)
        return ProductState(oostate.object_states)

class ArmTransitionModel(pomdp_py.TransitionModel):
    """ Static
    """
    def __init__(self, epsilon=1e-9):
        self._epsilon = epsilon
        super().__init__()

    def probability(self, next_state, state, action):
        if isinstance(action, Select):
            # new state?
            return 0.5
        if next_state != state:
            return self._epsilon
        return 1 - self._epsilon

    def sample(self, state, action):
        if isinstance(action, Select):
            return np.random.beta(1, 1)
            #return np.random.uniform(0, 1)
        return state
        #return deepcopy(state)
        
    def get_all_states(self):
        raise NotImplementedError

class AgentTransitionModel(pomdp_py.TransitionModel):
    def __init__(self):
        raise NotImplementedError

