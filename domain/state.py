
import pomdp_py

# STATES
GO = 0
STOP = 1

class ArmState(pomdp_py.ObjectState):
    def __init__(
        self,
        id: int,
        prob: float,
        shape: str, color: str, xy: tuple[float, float],
    ):
        super().__init__("arm", {
            "id": id,
            "prob": prob,
            "shape": shape,
            "color": color,
            "xy": xy,
        })

    def __str__(self):
        return f"Dot {self.xy} {self.shape} {self.color}"

class AgentState(pomdp_py.ObjectState):
    def __init__(self, id: int, state: str):
        super().__init__()
        self.id = id
        self.state = state

    def __str__(self):
        return f"Agent {self.id} ({self.state})"

class Go(AgentState):
    def __init__(self, id):
        super().__init__(id, "go")

class Stop(AgentState):
    def __init__(self, id):
        super().__init__(id, "stop")

class ProductState(pomdp_py.OOState):
    def __init__(self, arm_states):
        super().__init__(arm_states)
