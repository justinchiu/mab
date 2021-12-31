import pomdp_py

from agent.agent import RsAgent
from env.env import RsEnvironment

class RankingAndSelectionProblem(pomdp_py.OOPOMDP):
    def __init__(self):
        agent = RsAgent()
        env = RsEnvironment()
        super().__init__(agent, env, "RsPomdp")

### Solve the problem with POUCT/POMCP planner ###
### This is the main online POMDP solver logic ###
def solve(
    problem,
    max_depth=10,  # planning horizon
    discount_factor=0.99,
    planning_time=1.,       # amount of time (s) to plan each step
    exploration_const=1000, # exploration constant
    visualize=True,
    max_time=120,  # maximum amount of time allowed to solve the problem
    max_steps=500,
): # maximum number of planning steps the agent can take.
    """
    This function terminates when:
    - maximum time (max_time) reached; This time includes planning and updates
    - agent has planned `max_steps` number of steps
    - agent has taken n FindAction(s) where n = number of target objects.
    Args:
        visualize (bool) if True, show the pygame visualization.
    """

    random_objid = random.sample(problem.env.target_objects, 1)[0]
    random_object_belief = problem.agent.belief.object_beliefs[random_objid]
    if isinstance(random_object_belief, pomdp_py.Histogram):
        # Use POUCT
        planner = pomdp_py.POUCT(max_depth=max_depth,
                                 discount_factor=discount_factor,
                                 planning_time=planning_time,
                                 exploration_const=exploration_const,
                                 rollout_policy=problem.agent.policy_model)  # Random by default
    elif isinstance(random_object_belief, pomdp_py.Particles):
        # Use POMCP
        planner = pomdp_py.POMCP(max_depth=max_depth,
                                 discount_factor=discount_factor,
                                 planning_time=planning_time,
                                 exploration_const=exploration_const,
                                 rollout_policy=problem.agent.policy_model)  # Random by default
    else:
        raise ValueError("Unsupported object belief type %s" % str(type(random_object_belief)))

    robot_id = problem.agent.robot_id
    if visualize:
        viz = MosViz(problem.env, controllable=False)  # controllable=False means no keyboard control.
        if viz.on_init() == False:
            raise Exception("Environment failed to initialize")
        viz.update(robot_id,
                   None,
                   None,
                   None,
                   problem.agent.cur_belief)
        viz.on_render()

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
        reward = problem.env.state_transition(real_action, execute=True,
                                              robot_id=robot_id)

        # Receive observation
        _start = time.time()
        real_observation = \
            problem.env.provide_observation(problem.agent.observation_model, real_action)

        # Updates
        problem.agent.clear_history()  # truncate history
        problem.agent.update_history(real_action, real_observation)
        belief_update(problem.agent, real_action, real_observation,
                      problem.env.state.object_states[robot_id],
                      planner)
        _time_used += time.time() - _start

        # Info and render
        _total_reward += reward
        if isinstance(real_action, FindAction):
            _find_actions_count += 1
        print("==== Step %d ====" % (i+1))
        print("Action: %s" % str(real_action))
        print("Observation: %s" % str(real_observation))
        print("Reward: %s" % str(reward))
        print("Reward (Cumulative): %s" % str(_total_reward))
        print("Find Actions Count: %d" %  _find_actions_count)
        if isinstance(planner, pomdp_py.POUCT):
            print("__num_sims__: %d" % planner.last_num_sims)

        if visualize:
            # This is used to show the sensing range; Not sampled
            # according to observation model.
            robot_pose = problem.env.state.object_states[robot_id].pose
            viz_observation = MosOOObservation({})
            if isinstance(real_action, LookAction) or isinstance(real_action, FindAction):
                viz_observation = \
                    problem.env.sensors[robot_id].observe(robot_pose,
                                                          problem.env.state)
            viz.update(robot_id,
                       real_action,
                       real_observation,
                       viz_observation,
                       problem.agent.cur_belief)
            viz.on_loop()
            viz.on_render()

        # Termination check
        if set(problem.env.state.object_states[robot_id].objects_found)\
           == problem.env.target_objects:
            print("Done!")
            break
        if _find_actions_count >= len(problem.env.target_objects):
            print("FindAction limit reached.")
            break
        if _time_used > max_time:
            print("Maximum time reached.")
            break
