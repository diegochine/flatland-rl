from typing import List, Optional, Dict
import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.agent_utils import EnvAgent, RailAgentStatus
from flatland.envs.distance_map import DistanceMap
from flatland.envs.malfunction_generators import no_malfunction_generator
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_generators import random_rail_generator, RailGenerator
from flatland.envs.schedule_generators import random_schedule_generator, ScheduleGenerator


class RailEnvWrapper(RailEnv):
    """
    RailEnv environment class.

    RailEnv is an environment inspired by a (simplified version of) a rail
    network, in which agents (trains) have to navigate to their target
    locations in the shortest time possible, while at the same time cooperating
    to avoid bottlenecks.

    The valid actions in the environment are:

     -   0: do nothing (continue moving or stay still)
     -   1: turn left at switch and move to the next cell; if the agent was not moving, movement is started
     -   2: move to the next cell in front of the agent; if the agent was not moving, movement is started
     -   3: turn right at switch and move to the next cell; if the agent was not moving, movement is started
     -   4: stop moving

    Moving forward in a dead-end cell makes the agent turn 180 degrees and step
    to the cell it came from.


    The actions of the agents are executed in order of their handle to prevent
    deadlocks and to allow them to learn relative priorities.

    Reward Function:

    It costs each agent a step_penalty for every time-step taken in the environment. Independent of the movement
    of the agent. Currently all other penalties such as penalty for stopping, starting and invalid actions are set to 0.

    alpha = 1
    beta = 1
    Reward function parameters:

    - invalid_action_penalty = 0
    - step_penalty = -alpha
    - global_reward = beta
    - epsilon = avoid rounding errors
    - stop_penalty = 0  # penalty for stopping a moving agent
    - start_penalty = 0  # penalty for starting a stopped agent

    Stochastic malfunctioning of trains:
    Trains in RailEnv can malfunction if they are halted too often (either by their own choice or because an invalid
    action or cell is selected.

    Every time an agent stops, an agent has a certain probability of malfunctioning. Malfunctions of trains follow a
    poisson process with a certain rate. Not all trains will be affected by malfunctions during episodes to keep
    complexity managable.
    """
    alpha = 1
    beta = 2
    gamma = 1
    # Epsilon to avoid rounding errors
    epsilon = 0.01
    invalid_action_penalty = 0
    step_penalty = -1 * alpha
    global_reward = 1 * beta
    stop_penalty = 0.2  # penalty for stopping a moving agent
    start_penalty = 0.5  # penalty for starting a stopped agent

    reducing_distance_step = 1 * gamma
    deadlock_penalty = -3
    not_moving_penalty = 0.8

    def __init__(self,
                 width,
                 height,
                 rail_generator: RailGenerator = random_rail_generator(),
                 schedule_generator: ScheduleGenerator = random_schedule_generator(),
                 number_of_agents=1,
                 obs_builder_object: ObservationBuilder = GlobalObsForRailEnv(),
                 malfunction_generator_and_process_data=no_malfunction_generator(),
                 remove_agents_at_target=True,
                 random_seed=1,
                 record_steps=False
                 ):
        """
        Environment init.

        Parameters
        ----------
        rail_generator : function
            The rail_generator function is a function that takes the width,
            height and agents handles of a  rail environment, along with the number of times
            the env has been reset, and returns a GridTransitionMap object and a list of
            starting positions, targets, and initial orientations for agent handle.
            The rail_generator can pass a distance map in the hints or information for specific schedule_generators.
            Implementations can be found in flatland/envs/rail_generators.py
        schedule_generator : function
            The schedule_generator function is a function that takes the grid, the number of agents and optional hints
            and returns a list of starting positions, targets, initial orientations and speed for all agent handles.
            Implementations can be found in flatland/envs/schedule_generators.py
        width : int
            The width of the rail map. Potentially in the future,
            a range of widths to sample from.
        height : int
            The height of the rail map. Potentially in the future,
            a range of heights to sample from.
        number_of_agents : int
            Number of agents to spawn on the map. Potentially in the future,
            a range of number of agents to sample from.
        obs_builder_object: ObservationBuilder object
            ObservationBuilder-derived object that takes builds observation
            vectors for each agent.
        remove_agents_at_target : bool
            If remove_agents_at_target is set to true then the agents will be removed by placing to
            RailEnv.DEPOT_POSITION when the agent has reach it's target position.
        random_seed : int or None
            if None, then its ignored, else the random generators are seeded with this number to ensure
            that stochastic operations are replicable across multiple operations
        """
        super(RailEnv, self).__init__()

        # Previous distance intialization
        self.previous_distance = [400] * number_of_agents

        self.deadlocks = [False] * number_of_agents

        # Wait before Deadlock
        self.wait_deadlock = [0] * number_of_agents

        self.malfunction_generator, self.malfunction_process_data = malfunction_generator_and_process_data
        self.rail_generator: RailGenerator = rail_generator
        self.schedule_generator: ScheduleGenerator = schedule_generator
        self.rail: Optional[GridTransitionMap] = None
        self.width = width
        self.height = height

        self.remove_agents_at_target = remove_agents_at_target

        self.rewards = [0] * number_of_agents
        self.done = False
        self.obs_builder = obs_builder_object
        self.obs_builder.set_env(self)

        self._max_episode_steps: Optional[int] = None
        self._elapsed_steps = 0

        self.dones = dict.fromkeys(list(range(number_of_agents)) + ["__all__"], False)

        self.obs_dict = {}
        self.rewards_dict = {}
        self.dev_obs_dict = {}
        self.dev_pred_dict = {}

        self.agents: List[EnvAgent] = []
        self.number_of_agents = number_of_agents
        self.num_resets = 0
        self.distance_map = DistanceMap(self.agents, self.height, self.width)

        self.action_space = [5]

        self._seed()
        self._seed()
        self.random_seed = random_seed
        if self.random_seed:
            self._seed(seed=random_seed)

        self.valid_positions = None

        # global numpy array of agents position, True means that there is an agent at that cell
        self.agent_positions: np.ndarray = np.full((height, width), False)

        # save episode timesteps ie agent positions, orientations.  (not yet actions / observations)
        self.record_steps = record_steps  # whether to save timesteps
        self.cur_episode = []  # save timesteps in here

    @staticmethod
    def compute_max_episode_steps(width: int, height: int, ratio_nr_agents_to_nr_cities: float = 20.0) -> int:
        """
        compute_max_episode_steps(width, height, ratio_nr_agents_to_nr_cities, timedelay_factor, alpha)

        The method computes the max number of episode steps allowed

        Parameters
        ----------
        width : int
            width of environment
        height : int
            height of environment
        ratio_nr_agents_to_nr_cities : float, optional
            number_of_agents/number_of_cities

        Returns
        -------
        max_episode_steps: int
            maximum number of episode steps

        """
        timedelay_factor = 4
        alpha = 2
        return int(timedelay_factor * alpha * (width + height + ratio_nr_agents_to_nr_cities))

    def reset(self, regenerate_rail: bool = True, regenerate_schedule: bool = True, activate_agents: bool = False,
              random_seed: bool = None) -> (Dict, Dict):
        """
        reset(regenerate_rail, regenerate_schedule, activate_agents, random_seed)

        The method resets the rail environment

        Parameters
        ----------
        regenerate_rail : bool, optional
            regenerate the rails
        regenerate_schedule : bool, optional
            regenerate the schedule and the static agents
        activate_agents : bool, optional
            activate the agents
        random_seed : bool, optional
            random seed for environment

        Returns
        -------
        observation_dict: Dict
            Dictionary with an observation for each agent
        info_dict: Dict with agent specific information

        """

        if random_seed:
            self._seed(random_seed)

        optionals = {}
        if regenerate_rail or self.rail is None:
            rail, optionals = self.rail_generator(self.width, self.height, self.number_of_agents, self.num_resets,
                                                  np_random=np.random.RandomState())

            self.rail = rail
            self.height, self.width = self.rail.grid.shape

            # Do a new set_env call on the obs_builder to ensure
            # that obs_builder specific instantiations are made according to the
            # specifications of the current environment : like width, height, etc
            self.obs_builder.set_env(self)

        if optionals and 'distance_map' in optionals:
            self.distance_map.set(optionals['distance_map'])

        if regenerate_schedule or regenerate_rail or self.get_num_agents() == 0:
            agents_hints = None
            if optionals and 'agents_hints' in optionals:
                agents_hints = optionals['agents_hints']

            schedule = self.schedule_generator(self.rail, self.number_of_agents, agents_hints, self.num_resets,
                                               np_random=np.random.RandomState())
            self.agents = EnvAgent.from_schedule(schedule)

            if agents_hints and 'city_orientations' in agents_hints:
                ratio_nr_agents_to_nr_cities = self.get_num_agents() / len(agents_hints['city_orientations'])
                self._max_episode_steps = self.compute_max_episode_steps(
                    width=self.width, height=self.height,
                    ratio_nr_agents_to_nr_cities=ratio_nr_agents_to_nr_cities)
            else:
                self._max_episode_steps = self.compute_max_episode_steps(width=self.width, height=self.height)

        self.agent_positions = np.zeros((self.height, self.width), dtype=int) - 1

        self.reset_agents()

        for agent in self.agents:
            # Induce malfunctions
            if activate_agents:
                self.set_agent_active(agent)

            self._break_agent(agent)

            if agent.malfunction_data["malfunction"] > 0:
                agent.speed_data['transition_action_on_cellexit'] = RailEnvActions.DO_NOTHING

            # Fix agents that finished their malfunction
            self._fix_agent_after_malfunction(agent)

        self.num_resets += 1
        self._elapsed_steps = 0

        self.dones = dict.fromkeys(list(range(self.get_num_agents())) + ["__all__"], False)

        # Reset the state of the observation builder with the new environment
        self.obs_builder.reset()
        self.distance_map.reset(self.agents, self.rail)
        self.deadlocks = [False] * self.number_of_agents
        self.previous_distance = [np.linalg.norm(np.asarray(a.position) - np.asarray(a.target)) for a in self.agents]
        info_dict: Dict = {
            'action_required': {i: self.action_required(agent) for i, agent in enumerate(self.agents)},
            'malfunction': {
                i: agent.malfunction_data['malfunction'] for i, agent in enumerate(self.agents)
            },
            'speed': {i: agent.speed_data['speed'] for i, agent in enumerate(self.agents)},
            'status': {i: agent.status for i, agent in enumerate(self.agents)},
            'deadlock': {i: False for i, agent in enumerate(self.agents)}
        }
        # Return the new observation vectors for each agent
        observation_dict: Dict = self._get_observations()
        return observation_dict, info_dict

    def step(self, action_dict_: Dict[int, RailEnvActions]):
        """
        Updates rewards for the agents at a step.

        Parameters
        ----------
        action_dict_ : Dict[int,RailEnvActions]

        """
        self._elapsed_steps += 1

        # If we're done, set reward and info_dict and step() is done.
        if self.dones["__all__"]:
            self.rewards_dict = {}
            info_dict = {
                "action_required": {},
                "malfunction": {},
                "speed": {},
                "status": {},
                "deadlock": {}
            }
            for i_agent, agent in enumerate(self.agents):
                self.rewards_dict[i_agent] = self.global_reward if not self.deadlocks[
                    i_agent] else self.deadlock_penalty
                info_dict["action_required"][i_agent] = False
                info_dict["malfunction"][i_agent] = 0
                info_dict["speed"][i_agent] = 0
                info_dict["status"][i_agent] = agent.status
                info_dict["deadlock"][i_agent] = self.deadlocks[i_agent]

            return self._get_observations(), self.rewards_dict, self.dones, info_dict

        # Reset the step rewards
        self.rewards_dict = dict()
        info_dict = {
            "action_required": {},
            "malfunction": {},
            "speed": {},
            "status": {},
            "deadlock": {}
        }
        have_all_agents_ended = True  # boolean flag to check if all agents are done

        for i_agent, agent in enumerate(self.agents):
            # Reset the step rewards
            self.rewards_dict[i_agent] = 0

            # Induce malfunction before we do a step, thus a broken agent can't move in this step
            self._break_agent(agent)

            # Perform step on the agent
            self._step_agent(i_agent, action_dict_.get(i_agent))

            # manage the boolean flag to check if all agents are indeed done (or done_removed)
            have_all_agents_ended &= (agent.status in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED])

            # Build info dict
            info_dict["action_required"][i_agent] = self.action_required(agent)
            info_dict["malfunction"][i_agent] = agent.malfunction_data['malfunction']
            info_dict["speed"][i_agent] = agent.speed_data['speed']
            info_dict["status"][i_agent] = agent.status
            info_dict["deadlock"][i_agent] = self.deadlocks[i_agent]

            # Fix agents that finished their malfunction such that they can perform an action in the next step
            self._fix_agent_after_malfunction(agent)

        # Check for end of episode + set global reward to all rewards!
        if have_all_agents_ended:
            self.dones["__all__"] = True
            self.rewards_dict = {i: self.global_reward for i in range(self.get_num_agents())}
        if (self._max_episode_steps is not None) and (self._elapsed_steps >= self._max_episode_steps):
            self.dones["__all__"] = True
            for i_agent in range(self.get_num_agents()):
                self.dones[i_agent] = True
        if self.record_steps:
            self.record_timestep(action_dict_)

        return self._get_observations(), self.rewards_dict, self.dones, info_dict

    def _step_agent(self, i_agent, action: Optional[RailEnvActions] = None):
        """
        Performs a step and step, start and stop penalty on a single agent in the following sub steps:
        - malfunction
        - action handling if at the beginning of cell
        - movement

        Parameters
        ----------
        i_agent : int
        action_dict_ : Dict[int,RailEnvActions]

        """
        agent = self.agents[i_agent]
        if agent.status in [RailAgentStatus.DONE, RailAgentStatus.DONE_REMOVED]:  # this agent has already completed...
            return

        # agent gets active by a MOVE_* action and if c
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            if action in [RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_RIGHT,
                          RailEnvActions.MOVE_FORWARD] and self.cell_free(agent.initial_position):
                agent.status = RailAgentStatus.ACTIVE
                self._set_agent_to_initial_position(agent, agent.initial_position)
                self.rewards_dict[i_agent] += self.step_penalty * agent.speed_data['speed']
                return
            else:
                self.rewards_dict[i_agent] += self.step_penalty * agent.speed_data['speed']
                return

        agent.old_direction = agent.direction
        agent.old_position = agent.position

        # if agent is broken, actions are ignored and agent does not move.
        # full step penalty in this case
        if agent.malfunction_data['malfunction'] > 0:
            self.rewards_dict[i_agent] += self.step_penalty * agent.speed_data['speed']
            return

        # Is the agent at the beginning of the cell? Then, it can take an action.
        # As long as the agent is malfunctioning or stopped at the beginning of the cell,
        # different actions may be taken!
        if np.isclose(agent.speed_data['position_fraction'], 0.0, rtol=1e-03):
            # No action has been supplied for this agent -> set DO_NOTHING as default
            if action is None:
                action = RailEnvActions.DO_NOTHING

            if action < 0 or action > len(RailEnvActions):
                print('ERROR: illegal action=', action,
                      'for agent with index=', i_agent,
                      '"DO NOTHING" will be executed instead')
                action = RailEnvActions.DO_NOTHING

            if action == RailEnvActions.DO_NOTHING and agent.moving:
                # Keep moving
                action = RailEnvActions.MOVE_FORWARD

            if action == RailEnvActions.STOP_MOVING and agent.moving:
                # Only allow halting an agent on entering new cells.
                agent.moving = False
                agent_possible_transitions = self.rail.get_transitions(agent.position[0], agent.position[1],
                                                                       agent.direction)
                agent_num_transitions = np.count_nonzero(agent_possible_transitions)
                self.rewards_dict[i_agent] += self.stop_penalty

            if not agent.moving and not (
                    action == RailEnvActions.DO_NOTHING or
                    action == RailEnvActions.STOP_MOVING):
                # Allow agent to start with any forward or direction action
                agent.moving = True
                self.rewards_dict[i_agent] += self.start_penalty

            # Store the action if action is moving
            # If not moving, the action will be stored when the agent starts moving again.
            if agent.moving:
                _action_stored = False
                _, new_cell_valid, new_direction, new_position, transition_valid = \
                    self._check_action_on_agent(action, agent)

                if all([new_cell_valid, transition_valid]):
                    agent.speed_data['transition_action_on_cellexit'] = action
                    _action_stored = True
                else:
                    # But, if the chosen invalid action was LEFT/RIGHT, and the agent is moving,
                    # try to keep moving forward!
                    if (action == RailEnvActions.MOVE_LEFT or action == RailEnvActions.MOVE_RIGHT):
                        _, new_cell_valid, new_direction, new_position, transition_valid = \
                            self._check_action_on_agent(RailEnvActions.MOVE_FORWARD, agent)

                        if all([new_cell_valid, transition_valid]):
                            agent.speed_data['transition_action_on_cellexit'] = RailEnvActions.MOVE_FORWARD
                            _action_stored = True

                if not _action_stored:
                    # If the agent cannot move due to an invalid transition, we set its state to not moving
                    self.rewards_dict[i_agent] += self.invalid_action_penalty
                    self.rewards_dict[i_agent] += self.stop_penalty
                    agent.moving = False

        # Now perform a movement.
        # If agent.moving, increment the position_fraction by the speed of the agent
        # If the new position fraction is >= 1, reset to 0, and perform the stored
        #   transition_action_on_cellexit if the cell is free.
        if agent.moving:
            agent.speed_data['position_fraction'] += agent.speed_data['speed']
            if agent.speed_data['position_fraction'] > 1.0 or np.isclose(agent.speed_data['position_fraction'], 1.0,
                                                                         rtol=1e-03):
                # Perform stored action to transition to the next cell as soon as cell is free
                # Notice that we've already checked new_cell_valid and transition valid when we stored the action,
                # so we only have to check cell_free now!

                # cell and transition validity was checked when we stored transition_action_on_cellexit!
                cell_free, new_cell_valid, new_direction, new_position, transition_valid = self._check_action_on_agent(
                    agent.speed_data['transition_action_on_cellexit'], agent)

                # N.B. validity of new_cell and transition should have been verified before the action was stored!
                assert new_cell_valid
                assert transition_valid
                if cell_free:
                    self._move_agent_to_new_position(agent, new_position)
                    agent.direction = new_direction
                    agent.speed_data['position_fraction'] = 0.0
                    malfunction = 0
                    self.deadlocks[i_agent] = False
                    self.wait_deadlock[i_agent] = 0
                else:
                    self.wait_deadlock[i_agent] += 1
                    # for i in range(self.get_num_agents()):
                    #     if self.agents[i].position == new_position:
                    #         malfunction = self.agents[i].malfunction_data['malfunction'] > 0

                    if self.wait_deadlock[i_agent] >= 2:  # and (not malfunction):
                        self.rewards_dict[i_agent] = self.deadlock_penalty
                        self.deadlocks[i_agent] = True

            # has the agent reached its target?
            if np.equal(agent.position, agent.target).all():
                agent.status = RailAgentStatus.DONE
                self.dones[i_agent] = True
                self.active_agents.remove(i_agent)
                agent.moving = False
                self._remove_agent_from_scene(agent)
            else:
                # if the agent is reducing the distance from the target, i.e. it is getting closer, it is penalised less
                if (np.linalg.norm(np.asarray(agent.position) - np.asarray(agent.target)) <= self.previous_distance[i_agent]) \
                        and (self.reducing_distance_step != 0):
                    self.rewards_dict[i_agent] += self.reducing_distance_step * agent.speed_data['speed']
                else:
                    self.rewards_dict[i_agent] += self.step_penalty * agent.speed_data['speed']

                self.previous_distance[i_agent] = np.linalg.norm(np.asarray(agent.position) - np.asarray(agent.target))
        else:
            # step penalty if not moving (stopped now or before)
            self.rewards_dict[i_agent] += self.step_penalty * agent.speed_data['speed']
            # Additional penalty for not moving
            self.rewards_dict[i_agent] += self.not_moving_penalty * agent.speed_data['speed']
