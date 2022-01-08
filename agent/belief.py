
import numpy as np

import pomdp_py

from domain.state import ArmState, ProductState, Go, Stop, CountdownState

class ProductBelief(pomdp_py.OOBelief):
    def __init__(self, arm_beliefs):
        super().__init__(arm_beliefs)

    def mpe(self, **kwargs):
        return ProductState(pomdp_py.OOBelief.mpe(self, **kwargs).object_states)

    def random(self, **kwargs):
        return ProductState(pomdp_py.OOBelief.random(self, **kwargs).object_states)


def initialize_belief(
    num_dots,
    max_turns,
    prior,
    representation="histogram",
    num_particles=100,
    num_bins=10,
):
    """
    Returns a GenerativeDistribution that is the belief representation for
    ranking and selection / best arm identification in the multi-armed bandit problem.
    Args:
        num_dots (int): number of dots to choose between.
        prior (list[float]): A mapping [arm_id -> True -> [0,1]]. Use it!
        num_particles (int): Maximum number of particles used to represent the belief
    Returns:
        GenerativeDistribution: the initial belief representation.
    """
    if representation == "histogram":
        return _initialize_histogram_belief(
            num_dots, max_turns, prior, num_bins=num_bins,
        )
    elif representation == "particles":
        return _initialize_particles_belief(
            num_dots, max_turns, prior,
            num_particles = num_particles,
            num_bins = num_bins,
        )
    else:
        raise ValueError("Unsupported belief representation %s" % representation)

    
def _initialize_histogram_belief(dim, max_turns, prior, num_bins):
    """
    Returns the belief distribution represented as a histogram
    """
    if prior is None:
        prior = [[1. / num_bins for _ in range(num_bins)] for _ in range(dim)]
    oo_hists = {}  # objid -> Histogram
    # prior should be a tensor of shape num_dots (prob True)
    oo_hists = {
        id: pomdp_py.Histogram({
            ArmState(id, p, None, None, None): prob
            for p, prob in zip(np.linspace(.01, .99, num_bins), probs)
        })
        for (id, probs) in enumerate(prior)
    }
    agent_states = {
        Stop(id): .01
        for id in range(dim)
    }
    agent_states[Go()] = 1. - dim * .01

    counter_states = {
        CountdownState(max_turns): 1
    }
    oo_hists[dim] = pomdp_py.Histogram(agent_states)
    oo_hists[dim+1] = pomdp_py.Histogram(counter_states)

    # TODO: swap to numpy array, dict is super slow?
    return ProductBelief(oo_hists)


def sample_state(prior, dim, max_turns, robot_id, countdown_id):
    product_state = {}
    for id in range(dim):
        # sample an arm state
        dist = prior[id]
        prob = np.random.choice(dist)
        product_state[id] = ArmState(
            id, prob,
            shape= "large", color = "grey", xy = (1,1),
        )
    product_state[robot_id] = Go()
    product_state[countdown_id] = CountdownState(max_turns)
    return ProductState(product_state)


def _initialize_particles_belief(
    dim, max_turns, prior = None,
    num_particles = 100,
    num_bins = 5,
):
    """This returns a single set of particles that represent the distribution over a
    joint state space of all objects.
    Since it is very difficult to provide a prior knowledge over the joint state
    space when the number of objects scales, the prior (which is
    object-oriented), is used to create particles separately for each object to
    satisfy the prior; That is, particles beliefs are generated for each object
    as if object_oriented=True. Then, `num_particles` number of particles with
    joint state is sampled randomly from these particle beliefs.
    """
    # prior: Dict(arm_id -> List[probability of success])

    if prior is None:
        # uniform over discretization
        prior = {id: np.linspace(.01, .99, num_bins) for id in range(dim)}
        # alternative: sample from uniform / beta distribution

    particles = [
        sample_state(prior, dim, max_turns, dim, dim+1)
        for _ in range(num_particles)
    ]
    return pomdp_py.Particles(particles)
