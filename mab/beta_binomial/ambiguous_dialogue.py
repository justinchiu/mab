"""
In this problem we test whether models can deal with ambiguity.
Policy needs to select the unambiguous dot.
"""

from typing import Optional
from itertools import product

import numpy as np
import pomdp_py
from pomdp_py.utils.debugging import TreeDebugger

from problem import RankingAndSelectionProblem, belief_update
from mab.beta_bernoulli.domain.action import Ask, Select, Pass
from mab.beta_bernoulli.domain.observation import ProductObservation


max_turns = 5
total_dots = 5
num_dots = 3
num_targets = 1

def initialize_dots():
    bs = ("black", "small")
    gl = ("grey", "large")
    rm = ("red", "medium")

    dot_vector_A = np.array([1, 2, 0])
    dot_vector_B = np.array([1, 1, 1])
    attributes_A = [bs, gl, rm]
    attributes_B = [bs, gl, gl]
    return dot_vector_A, dot_vector_B, attributes_A, attributes_B


dots_A, dots_B, attrs_A, attrs_B  = initialize_dots()
print("dots A")
print(dots_A)
print(attrs_A)
print("dots B")
print(dots_B)
print(attrs_B)

problems = [RankingAndSelectionProblem(
    dots,
    max_turns,
    belief_rep = "particles",
    num_bins=2,
    num_particles = 0,
    enumerate_belief = True,
) for dots in (dots_A, dots_B)]

planner = pomdp_py.POMCP(
    max_depth = max_turns, # need to change
    discount_factor = 1,
    num_sims = 10000,
    exploration_const = 100,
    #rollout_policy = problems[0].agent.policy_model, # need to change per agent?
)

def plan(planner, problem, steps_left) -> pomdp_py.Action:
    planner.clear_agent()
    planner.set_max_depth(steps_left)
    planner.set_rollout_policy(problem.agent.policy_model)
    action = planner.plan(problem.agent)
    return action

# communicate by passing boolean functions that apply to the dot you mention

def observe(action_A, attrs_A, attrs_B) -> np.array:
    if not isinstance(action_A, Ask):
        return False
    lenA = len(attrs_A)
    lenB = len(attrs_B)
    # whether player B can see anything with the attributes of player A's ask
    # give back to player A
    response_from_B = np.zeros_like(action_A.val, dtype=int)
    # the ask gives information about dots A has
    # use to update player B's belief
    observation_for_B = np.zeros(len(attrs_B), dtype=int)
    for i, (attr, present) in enumerate(zip(attrs_A, action_A.val)):
        if present:
            # for each nonzero value of action_A, respond with whether B can see that attribute
            for j, attrB in enumerate(attrs_B):
                if attrB == attr:
                    observation_for_B[j] += 1
                    response_from_B[i] += 1
    return response_from_B, observation_for_B

def force_expansion(planner, agent, action, obs, steps_left):
    planner.clear_agent()
    planner._agent = agent
    planner.set_max_depth(steps_left)
    planner.set_rollout_policy(agent.policy_model)
    planner.force_expansion(action, obs)

"""
On the first turn, player A and B must initialize their belief trees.
Turns then proceed following

Player A                    Player B
Update belief from response_from_B
Respond to ask_B (player B first action = Pass)
Update belief from ask_B
Plan
Ask
                            Update belief from response_from_A
                            Respond 
                            Update belief if Asked about present dot
                            Plan
                            Ask

Simplification: Responses are sent separately from Asks
"""

def take_turn(
    id_A,
    turn,
    action_A,
    response_from_B,
    action_B,
    attrs_A, attrs_B,
    num_dots,
    max_turns,
):
    problem = problems[id_A]
    agent = problem.agent

    belief = agent.belief
    #import pdb; pdb.set_trace()

    # We are player A
    if response_from_B is not None and isinstance(action_A, Ask):
        # Process response from B to previous action_A
        #print(f"num particles {len(agent.tree.belief.particles)}")
        belief_update(
            agent, action_A, response_from_B,
            problem.env.state.object_states[agent.id],
            problem.env.state.object_states[agent.countdown_id],
            planner,
        )
        #print(f"num particles {len(agent.tree.belief.particles)}")

    if isinstance(action_B, Ask):
        # Respond if asked, only after first turn
        response_from_A, observation_for_A_vec = observe(action_B, attrs_B, attrs_A)
        # sigh, convert to dict
        response_from_A = ProductObservation({
            id: response_from_A[id]
        for id in range(response_from_A.shape[0])})

        # convert Ask from B to an Observation for A
        # WRONG
        observation_for_A = ProductObservation({
            id: observation_for_A_vec[id]
        for id in range(observation_for_A_vec.shape[0])})

        if observation_for_A_vec.sum() > 0:
            # create dummy action
            action_A0 = Ask(observation_for_A_vec)
            if agent.tree[action_A0][observation_for_A] is None:
                force_expansion(
                    planner, agent, action_A0, observation_for_A,
                    steps_left = max_turns - turn,
                )
            next_node = agent.tree[action_A0][observation_for_A]
            num_particles = len(next_node.belief.particles)
            #print(f"num particles {len(agent.tree.belief.particles)}")
            #print(f"num particles {num_particles}")
            # particle reinvigoration fails even though a node has been visited > 0 times.
            if num_particles == 0:
                # it's possible we have moved to a node that has not been expanded
                # replan to populate beliefs
                plan(planner, problems[id_A], steps_left = max_turns - turn)
            belief_update(
                agent, action_A0, observation_for_A,
                problem.env.state.object_states[agent.id],
                problem.env.state.object_states[agent.countdown_id],
                planner,
            )
            #import pdb; pdb.set_trace()
    elif isinstance(action_B, Select):
        response_from_A = None
        # must select
        pass
    else:
        response_from_A = None

    # Plan and take action
    action_A = plan(planner, problems[id_A], steps_left = max_turns - turn)
    reward_A = problems[id_A].env.state_transition(action_A, execute=True)

    return response_from_A, action_A, reward_A


# initialize trees for both agents
# not sure this is necessary
_ = plan(planner, problems[0], steps_left = max_turns)
_ = plan(planner, problems[1], steps_left = max_turns)

action_A = Pass()
action_B = Pass()
response_from_A = None
response_from_B = None
rA = 0
rB = 0
default_response = ProductObservation({id: 0 for id in range(num_dots)})
for turn in range(max_turns):
    # player A goes first
    response_from_A, action_A, reward_A = take_turn(
        id_A = 0,
        turn = turn,
        action_A = action_A,
        response_from_B = response_from_B,
        action_B = action_B,
        attrs_A = attrs_A, attrs_B = attrs_B,
        num_dots = num_dots,
        max_turns = max_turns,
    )

    print(f"Turn {turn}")
    if isinstance(action_B, Ask):
        print(f"Response A: {response_from_A}")
    print(f"Action A: {action_A}")

    response_from_B, action_B, reward_B = take_turn(
        id_A = 1,
        turn = turn,
        action_A = action_B,
        response_from_B = response_from_A,
        action_B = action_A,
        attrs_A = attrs_B, attrs_B = attrs_A,
        num_dots = num_dots,
        max_turns = max_turns,
    )
    if isinstance(action_A, Ask):
        print(f"Response B: {response_from_B}")
    print(f"Action B: {action_B}")


    rA += reward_A
    rB += reward_B
    if isinstance(action_A, Select):
        break

print(f"Total rewards: A ({rA}) B ({rB})")
