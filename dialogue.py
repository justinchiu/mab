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
    num_sims = 20000,
    exploration_const = 100,
    #rollout_policy = problems[0].agent.policy_model, # need to change per agent?
)

def plan(planner, problem, steps_left):
    planner.clear_agent()
    planner.set_max_depth(steps_left)
    planner.set_rollout_policy(problem.agent.policy_model)
    action = planner.plan(problem.agent)
    return action

# communicate by passing boolean functions that apply to the dot you mention

def observe_B(action_A, attrs_A, attrs_B):
    if not isinstance(action_A, Ask):
        return False
    # whether player B can see anything with the attributes of player A's ask
    feedback = np.zeros_like(action_A.val)
    for i, (attr, present) in enumerate(zip(attrs_A, action_A.val)):
        feedback[i] = present and attr in attrs_B
    return feedback

def map_obs(action_A, attrs_A, attrs_B):
    if not isinstance(action_A, Ask):
        return False
    # whether player B can see anything with the attributes of player A's ask
    observation = np.zeros_like(action_A.val)
    for i, attr in enumerate(attrs_B):
        if attr in attrs_A:
            observation[i] = action_A.val[attrs_A.index(attr)]
    return observation

action_B = Pass()
for turn in range(max_turns):
    # player A goes first
    action_A = plan(planner, problems[0], steps_left = max_turns - turn)
    reward_A = problems[0].env.state_transition(action_A, execute=True)

    print(f"Turn {turn}")
    print(f"Action A: {action_A}")

    # player B observes action_A and just responds (for now)
    response_from_B = observe_B(action_A, attrs_A, attrs_B)
    observation_for_A = ProductObservation({
        id: 1 if action_A.val[id] and response_from_B[id] else 0
    for id in range(num_dots)})
    # double check response
    real_observation = problems[0].env.provide_observation(
        problems[0].agent.observation_model, action_A)

    # player B observes player A's action
    observation_for_B_vec = map_obs(action_A, attrs_A, attrs_B)
    observation_for_B = ProductObservation({
        id: 1 if observation_for_B_vec[id] else 0
    for id in range(num_dots)})
    if observation_for_B_vec.sum() > 0:
        # only update belief if gained information
        # call plan to initialize tree for player B
        _ = plan(planner, problems[1], steps_left = max_turns - turn)
        action_B0 = Ask(obseration_for_B)
        belief_update(
            problems[1].agent, action_B0, observation_for_B,
            robot_state, countdown_state,
            planner,
        )

    # player B can also send more information
    action_B = plan(planner, problems[1], steps_left = max_turns - turn)
    reward_B = problems[1].env.state_transition(action_B, execute=True)

    # observation for A should be both response + action_B

    import pdb; pdb.set_trace()


