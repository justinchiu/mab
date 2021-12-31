
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
        super().__init__()
        self.id = id
        self.prob = prob
        self.shape = shape
        self.color = color
        self.xy = xy

    def __str__(self):
        return f"Dot {self.xy} {self.shape} {self.color}"

class AgentState(pomdp_py.ObjectState):
    # not sure this is necessary
    def __init__(self, id: int):
        super().__init__()
        self.id = id
        self.state = GO

    def __str__(self):
        return f"Agent {self.id}"
