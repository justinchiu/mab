
import pomdp_py

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
        })
        """
        super().__init__("arm", {
            "id": id,
            "prob": prob,
            "shape": shape,
            "color": color,
            "xy": xy,
        })
        """

    def __str__(self):
        return f"Dot {self.attributes['id']} ({self.attributes['prob']})"
        return f"Dot {self.xy} {self.shape} {self.color}"

class AgentState(pomdp_py.ObjectState):
    def __init__(self, state: str):
        super().__init__("agent", {"state": state})
        self.state = state

    def __str__(self):
        return f"Agent ({self.state})"

class Go(AgentState):
    def __init__(self):
        super().__init__("go")

class Stop(AgentState):
    def __init__(self, id):
        super().__init__("stop")
        self.id = id

    def __str__(self):
        return f"Agent ({self.state}: {self.id})"

class ProductState(pomdp_py.OOState):
    def __init__(self, arm_states):
        super().__init__(arm_states)
