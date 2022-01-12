# Multi-Armed Bandit POMDP
# pomdp-py implementation
# based off of pomdp_py/pomdp_problems/multi_object_search

import random
import time

from copy import deepcopy
from itertools import product

import numpy as np
import pomdp_py
from pomdp_py.utils import TreeDebugger

from agent.agent import RsAgent
from agent.belief import ProductBelief
from domain.action import Ask
from domain.state import ArmState, ProductState, Go, Stop, CountdownState
from env.env import RsEnvironment

DBG_OUTER = False
DBG_UPDATE = False


def initialize_dots(total_dots=9, num_dots=7, num_targets=4):
    # attributes
    color = ["black", "grey", "white"]
    size = ["large", "medium", "small"]
    attributes = list(product(color, size))

    all_dot_vector = np.zeros(total_dots, dtype=np.bool_)
    diff = total_dots - num_dots

    target_dots = np.random.choice(
        num_dots - diff, num_targets, replace=False)

    all_dot_vector[target_dots + diff] = True

    dot_vector_A = all_dot_vector[:num_dots]
    dot_vector_B = all_dot_vector[-num_dots:]
    attributes_A = attributes[:num_dots]
    attributes_B = attributes[-num_dots:]
    return dot_vector_A, dot_vector_B, attributes_A, attributes_B


class RankingAndSelectionProblem(pomdp_py.OOPOMDP):
    def __init__(
        self,
        dot_vector,
        max_turns,
        belief_rep="histogram", prior=None,
        num_particles=100,
        num_bins = 5,
    ):
        self.delta = 0.01
        self.dot_vector = dot_vector
        num_dots = dot_vector.shape[0]
        num_targets = dot_vector.sum()
        self.num_dots = num_dots
        self.num_targets = num_targets

        state = {
            id: ArmState(
                id, 1 - self.delta if is_good else self.delta,
                shape = "large", color = "grey", xy = (1,1),
            ) for id, is_good in enumerate(self.dot_vector)
        }
        state[num_dots] = Go()
        state[num_dots+1] = CountdownState(max_turns)
        init_true_state = ProductState(state)

        agent = RsAgent(num_dots, max_turns, belief_rep, prior, num_particles, num_bins)
        env = RsEnvironment(num_dots, self.dot_vector, init_true_state)
        super().__init__(agent, env, "RankingAndSelectionPomdp")


### Belief Update
# why is this here instead of in the particle filter?
# seems like this is only run for UCT, maybe stick to POMCP in general then?
### Belief Update ###
def belief_update(
    agent, real_action, real_observation,
    next_robot_state, next_countdown_state,
    planner,
):
    """Updates the agent's belief; The belief update may happen
    through planner update (e.g. when planner is POMCP)."""
    # Updates the planner; In case of POMCP, agent's belief is also updated.
    planner.update(agent, real_action, real_observation)

    # Update agent's belief, when planner is not POMCP
    if not isinstance(planner, pomdp_py.POMCP):
        # Update belief for every object
        for objid in agent.cur_belief.object_beliefs:
            belief_obj = agent.cur_belief.object_belief(objid)
            if DBG_UPDATE:
                print("action:")
                print(real_action)
                print("observation:")
                print(real_observation)
                print("next robot state")
                print(next_robot_state)
                print("old belief")
                print(belief_obj.histogram)
            if isinstance(belief_obj, pomdp_py.Histogram):
                if objid == agent.id:
                    # Assuming the agent can observe its own state:
                    new_belief = pomdp_py.Histogram({next_robot_state: 1.0})
                elif objid == agent.countdown_id:
                    new_belief = pomdp_py.Histogram({next_countdown_state: 1.0})
                elif isinstance(real_action, Ask) and real_action.val[objid]:
                    # This is doing
                    #    B(si') = normalizer * O(oi|si',sr',a) * sum_s T(si'|s,a)*B(si)
                    #
                    # Notes: First, objects are static; Second,
                    # O(oi|s',a) ~= O(oi|si',sr',a) according to the definition
                    # of the observation model in models/observation.py.  Note
                    # that the exact belief update rule for this OOPOMDP needs to use
                    # a model like O(oi|si',sr',a) because it's intractable to
                    # consider s' (that means all combinations of all object
                    # states must be iterated). 
                    new_belief = pomdp_py.update_histogram_belief(
                        belief_obj,
                        real_action,
                        real_observation.for_obj(objid),
                        agent.observation_model[objid],
                        agent.transition_model[objid],
                        # The agent knows the objects are static.
                        static_transition=objid != agent.id,
                        oargs={"next_robot_state": next_robot_state,
                    })
                    if DBG_UPDATE:
                        print("new belief")
                        print(new_belief)
                else:
                    # same
                    new_belief = deepcopy(belief_obj)
            else:
                raise ValueError("Unexpected program state."\
                                 "Are you using the appropriate belief representation?")

            agent.cur_belief.set_object_belief(objid, new_belief)

### Solve the problem with POUCT/POMCP planner ###
### This is the main online POMDP solver logic ###
def solve(
    problem,
    max_depth=5,  # planning horizon
    discount_factor=0.99,
    #planning_time=5.,       # amount of time (s) to plan each step
    num_sims=10000,
    exploration_const=1000, # exploration constant
    visualize=True,
    max_time=120,  # maximum amount of time allowed to solve the problem
    max_steps=5
): # maximum number of planning steps the agent can take.
    """
    This function terminates when:
    - maximum time (max_time) reached; This time includes planning and updates
    - agent has planned `max_steps` number of steps
    - agent has taken n FindAction(s) where n = number of target objects.
    Args:
        visualize (bool) if True, show the pygame visualization.
    """

    #random_objid = random.sample(problem.env.target_objects, 1)[0]
    #random_objid = random.choice(range(problem.num_dots))
    #random_object_belief = problem.agent.belief.object_beliefs[random_objid]
    belief = problem.agent.belief
    if isinstance(belief, ProductBelief):
        # HISTOGRAM
        # Use POUCT
        planner = pomdp_py.POUCT(max_depth=max_depth,
                                 discount_factor=discount_factor,
                                 #planning_time=planning_time,
                                 num_sims = num_sims,
                                 exploration_const=exploration_const,
                                 rollout_policy=problem.agent.policy_model)  # Random by default
    elif isinstance(belief, pomdp_py.Particles):
        # Use POMCP
        planner = pomdp_py.POMCP(max_depth=max_depth,
                                 discount_factor=discount_factor,
                                 #planning_time=planning_time,
                                 num_sims = num_sims,
                                 exploration_const=exploration_const,
                                 rollout_policy=problem.agent.policy_model)  # Random by default
    else:
        raise ValueError("Unsupported object belief type %s" % str(type(random_object_belief)))

    print(f"True state: {problem.env.dot_vector}")

    def print_belief():
        print(problem.agent.cur_belief.object_beliefs[0].histogram)
        print(problem.agent.cur_belief.object_beliefs[1].histogram)
        print(problem.agent.cur_belief.object_beliefs[2].histogram)
        print(problem.agent.cur_belief.object_beliefs[3].histogram)
        print(problem.agent.cur_belief.object_beliefs[4].histogram)
        print(problem.agent.cur_belief.object_beliefs[5].histogram)
        print(problem.agent.cur_belief.object_beliefs[6].histogram)
    if DBG_OUTER:
        print("initial belief")
        print_belief()

    _time_used = 0
    _find_actions_count = 0
    _total_reward = 0  # total, undiscounted reward
    for i in range(max_steps):
        # Plan action
        _start = time.time()
        real_action = planner.plan(problem.agent)
        _time_used += time.time() - _start
        if _time_used > max_time:
            break  # no more time to update.

        # Execute action
        reward = problem.env.state_transition(real_action, execute=True)

        # Receive observation
        _start = time.time()
        real_observation = problem.env.provide_observation(
            problem.agent.observation_model, real_action)

        # Updates
        # no need to truncate history here though
        # (used for POUCT)
        problem.agent.clear_history()  # truncate history
        problem.agent.update_history(real_action, real_observation)
        belief_update(
            problem.agent, real_action, real_observation,
            problem.env.state.object_states[problem.agent.id],
            problem.env.state.object_states[problem.agent.countdown_id],
            planner,
        )
        _time_used += time.time() - _start


        # Info and render
        _total_reward += reward
        if isinstance(real_action, Ask):
            _find_actions_count += 1
        print("==== Step %d ====" % (i+1))
        print("Action: %s" % str(real_action))
        print("Observation: %s" % str(real_observation))
        print("Reward: %s" % str(reward))
        print("Reward (Cumulative): %s" % str(_total_reward))
        print("Find Actions Count: %d" %  _find_actions_count)
        if isinstance(planner, pomdp_py.POUCT):
            print("__num_sims__: %d" % planner.last_num_sims)

        if DBG_OUTER:
            print("new belief")
            print_belief()
            import pdb; pdb.set_trace()

        # Termination check
        agent_state = problem.env.state.object_states[problem.agent.id]
        stopped = isinstance(agent_state, Stop)
        if stopped and problem.dot_vector[agent_state.id]:
            print("Success!")
            break
        if stopped and not problem.dot_vector[agent_state.id]:
            print("Failure")
            break
        if _time_used > max_time:
            print("Maximum time reached.")
            break

if __name__ == "__main__":
    total_dots = 7
    num_dots = 5
    num_targets = 2
    max_turns = num_dots
    dot_vector, _, _, _ = initialize_dots(total_dots, num_dots, num_targets)

    TEST_POUCT = False
    if TEST_POUCT:
        # Test POUCT
        problem = RankingAndSelectionProblem(
            dot_vector, max_turns,
            num_bins=3,
            belief_rep="histogram",
        )
        """
        planner = pomdp_py.POUCT(
            max_depth=5,
            discount_factor=1,
            num_sims = 20000,
            exploration_const=100,
            rollout_policy=problem.agent.policy_model)  # Random by default
        action = planner.plan(problem.agent)
        dd = TreeDebugger(problem.agent.tree)
        # Observation: needs an extremely large number of simulations if symmetry
        # is not taken advantage of.
        # Is POMCP better?
        """
        solve(problem, visualize=False, num_sims=50000)

    TEST_POMCP = True
    if TEST_POMCP:
        # Test POMCP
        problem = RankingAndSelectionProblem(
            dot_vector, max_turns,
            belief_rep = "particles",
            num_bins = 5,
            num_particles = 1000,
        )
        planner = pomdp_py.POMCP(
            max_depth=5,
            discount_factor=1,
            num_sims = 20000,
            exploration_const=100,
            rollout_policy=problem.agent.policy_model)  # Random by default
        action = planner.plan(problem.agent)
        dd = TreeDebugger(problem.agent.tree)
        #import pdb; pdb.set_trace()
        solve(problem, visualize=False, num_sims = 50000)
