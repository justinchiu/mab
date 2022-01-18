
from typing import Optional

import numpy as np
import pomdp_py

from problem import RankingAndSelectionProblem, initialize_dots, belief_update
from domain.action import Ask, Select, Pass
from domain.observation import ProductObservation


num_players = 2
max_turns = 5

total_dots = 7
num_dots = 5
num_targets = 2

dots_A, dots_B, attrs_A, attrs_B  = initialize_dots(total_dots, num_dots, num_targets)
print("dots A")
print(dots_A)
print("dots B")
print(dots_B)

problems = [RankingAndSelectionProblem(
    dots,
    max_turns,
    belief_rep = "particles",
    num_bins=5,
    num_particles = 2000,
) for dots in (dots_A, dots_B)]

planner = pomdp_py.POMCP(
    max_depth = max_turns, # need to change
    discount_factor = 1,
    num_sims = 1000,
    exploration_const = 200,
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
    # whether player B can see anything with the attributes of player A's ask
    # give back to player A
    response_from_B = np.zeros_like(action_A.val)
    # the ask gives information about dots A has
    # use to update player B's belief
    observation_for_B = np.zeros_like(action_A.val)
    for i, (attr, present) in enumerate(zip(attrs_A, action_A.val)):
        response_from_B[i] = present and attr in attrs_B
    for i, attr in enumerate(attrs_B):
        if attr in attrs_A:
            observation_for_B[i] = action_A.val[attrs_A.index(attr)]
    return response_from_B, observation_for_B

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
    # We are player A
    if response_from_B is not None and isinstance(action_A, Ask):
        # Process response from B to previous action_A
        belief_update(
            problems[id_A].agent, action_A, response_from_B,
            robot_state_A, countdown_state,
            planner,
        )

    if isinstance(action_B, Ask):
        # Respond if asked, only after first turn
        response_from_A, observation_for_A_vec = observe(action_B, attrs_B, attrs_A)
        response_from_A = ProductObservation({
            id: 1 if action_B.val[id] and response_from_A[id] else 0
        for id in range(num_dots)})

        # convert Ask from B to an Observation for A
        observation_for_A = ProductObservation({
            id: 1 if observation_for_A_vec[id] else 0
        for id in range(num_dots)})

        if observation_for_A_vec.sum() > 0:
            # create dummy action
            action_A0 = Ask(observation_for_A)
            belief_update(
                problems[0].agent, action_A0, observation_for_A,
                robot_state_A, countdown_state,
                planner,
            )
    elif isinstance(action_B, Select):
        response_from_A = None
        # must select
        pass
    else:
        response_from_A = None

    # Plan and ask
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
        max_turns = max_turns
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

    import pdb; pdb.set_trace()
