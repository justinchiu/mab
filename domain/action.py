
import pomdp_py

class Action(pomdp_py.Action):
    """Bandit action; Simple named action."""
    def __init__(self, name, val):
        self.name = name
        self.val = val
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, Action):
            return self.name == other.name and self.val == other.val
        elif type(other) == str:
            return self.name == other
    def __str__(self):
        return self.name
    def __repr__(self):
        return f"Action({self.name}): {self.val}"

class Ask(Action):
    def __init__(self, ids):
        super().__init__("ask", ids)

class Select(Action):
    def __init__(self, id):
        super().__init__("select", id)
