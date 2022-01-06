
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
        return _initialize_histogram_belief(num_dots, max_turns, prior, num_bins=num_bins)
    elif representation == "particles":
        return _initialize_particles_belief(num_dots, max_turns, prior, num_particles=num_particles)
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


def _initialize_particles_belief(
    dim, max_turns, prior, num_particles=100,
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
    # For the robot, we assume it can observe its own state;
    # Its pose must have been provided in the `prior`.
    raise NotImplementedError

    assert robot_id in prior, "Missing initial robot pose in prior."
    init_robot_pose = list(prior[robot_id].keys())[0]
    
    oo_particles = {}  # objid -> Particageles
    width, length = dim
    for objid in object_ids:
        particles = [RobotState(robot_id, init_robot_pose, (), None)]  # list of states; Starting the observable robot state.
        if objid in prior:
            # prior knowledge provided. Just use the prior knowledge
            prior_sum = sum(prior[objid][pose] for pose in prior[objid])
            for pose in prior[objid]:
                state = ObjectState(objid, "target", pose)
                amount_to_add = (prior[objid][pose] / prior_sum) * num_particles
                for _ in range(amount_to_add):
                    particles.append(state)
        else:
            # no prior knowledge. So uniformly sample `num_particles` number of states.
            for _ in range(num_particles):
                x = random.randrange(0, width)
                y = random.randrange(0, length)
                state = ObjectState(objid, "target", (x,y))
                particles.append(state)

        particles_belief = pomdp_py.Particles(particles)
        oo_particles[objid] = particles_belief
        
    # Return Particles distribution which contains particles
    # that represent joint object states
    particles = []
    for _ in range(num_particles):
        object_states = {}
        for objid in oo_particles:
            random_particle = random.sample(oo_particles[objid], 1)[0]
            object_states[_id] = copy.deepcopy(random_particle)
        particles.append(ProductState(object_states))
    return pomdp_py.Particles(particles)
