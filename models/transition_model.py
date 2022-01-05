import copy
import pomdp_py

from domain.action import Ask, Select
from domain.state import Go, Stop, ProductState

class ProductTransitionModel(pomdp_py.OOTransitionModel):
    def __init__(self, num_dots, epsilon=1e-9):
        self._epsilon = epsilon
        transition_models = {
            id: ArmTransitionModel(id) for id in range(1, num_dots+1)
        }
        transition_models[0] = AgentTransitionModel(0)
        super().__init__(transition_models)


    def sample(self, state, action, **kwargs):
        # odd this does not do map(models, lamdbda x: x.sample)
        product_state = pomdp_py.OOTransitionModel.sample(self, state, action, **kwargs)
        return ProductState(product_state.object_states)

class ArmTransitionModel(pomdp_py.TransitionModel):
    """ Static
    """
    def __init__(self, id, epsilon=1e-9):
        self.id = id
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
        return copy.deepcopy(state.object_states[self.id])
        if isinstance(action, Select):
            return np.random.beta(1, 1)
            #return np.random.uniform(0, 1)
        return state
        #return deepcopy(state)
        
    def get_all_states(self):
        raise NotImplementedError

class AgentTransitionModel(pomdp_py.TransitionModel):
    def __init__(self, id):
        super().__init__()
        self.id = id

    def probability(self, next_robot_state, next_state, action):
        pass

    def argmax(self, product_state, action):
        assert isinstance(product_state, ProductState)
        state = product_state.object_states[0]
        if  isinstance(state, Stop):
            # no change if already stopped
            return state
            #return product_state
        elif isinstance(action, Select):
            return Stop(action.val)
        elif isinstance(action, Ask):
            return Go()

    def sample(self, state, action):
        return self.argmax(state, action)
