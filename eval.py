import numpy as np
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnvActions

import wandb
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.utils.rendertools import RenderTool

from flatland_agent import FlatlandDQNAgent
from processor import normalize_observation
from rail_env import RailEnvWrapper


def flatland_test(width, height, action_shape, n_agents, tree_depth, max_num_cities=0, max_rails_between_cities=10, max_rails_in_city=2,
                  agent=None, path=None, n_episodes=10, max_steps=200, use_wandb=False):
    if max_num_cities == 0:
        max_num_cities = n_agents
    rail_generator = sparse_rail_generator(max_num_cities=max_num_cities,
                                           max_rails_between_cities=max_rails_between_cities,
                                           max_rails_in_city=max_rails_in_city,
                                           seed=48)
    tree_obs = TreeObsForRailEnv(max_depth=tree_depth, predictor=ShortestPathPredictorForRailEnv())
    env = RailEnvWrapper(width=width, height=height,
                         rail_generator=rail_generator,
                         number_of_agents=n_agents,
                         obs_builder_object=tree_obs,
                         random_seed=48)
    # env_renderer = RenderTool(env)

    if agent is None:
        if path is None:
            raise ValueError('Both path and agent are none.. ')
        else:
            player = FlatlandDQNAgent.load(path, 'final', training=False)
    else:
        player = agent

    scores = []
    arrival_scores = []
    deadlocks_scores = []

    # Empty dictionary for all agent action
    action_dict = dict()

    for episode in range(n_episodes):

        next_obs, info = env.reset()
        # env_renderer.reset()
        state = {a: normalize_observation(next_obs[a], tree_depth) for a in range(env.get_num_agents())}
        score = 0
        step = 0
        done = {'__all__': False}

        while not done['__all__'] and step < max_steps and not all(info['deadlock'].values()):
            # Chose an action for each agent in the environment
            for a in range(env.get_num_agents()):
                if info['action_required'][a]:
                    mask = np.full(action_shape, True)
                    if info["status"][a] == RailAgentStatus.ACTIVE:
                        for action in RailEnvActions:
                            mask[int(action)] = env._check_action_on_agent(action, env.agents[a])[-1]
                    action = player.act(state[a], mask=mask)
                else:
                    action = 0
                action_dict.update({a: action})

            next_obs, all_rewards, done, info = env.step(action_dict)

            # env_renderer.render_env(show=True, show_observations=True, show_predictions=False)

            next_state = {a: None for a in range(env.get_num_agents())}
            next_state.update({a: normalize_observation(next_obs[a], tree_depth)
                               for a in range(env.get_num_agents())
                               if next_obs[a] is not None})

            for a in range(env.get_num_agents()):
                score += all_rewards[a]
            state = next_state
            step += 1

        trains_arrived = sum([done[a] for a in range(n_agents)])
        trains_deadlocked = sum(info['deadlock'].values())
        arrival_scores.append(trains_arrived)
        deadlocks_scores.append(trains_deadlocked)
        scores.append(score)
    if use_wandb:
        wandb.log({'test': {'score': np.mean(scores),
                            'arrivals': np.mean(arrival_scores),
                            'deadlocks': np.mean(deadlocks_scores)}})

    # env_renderer.close_window()
    # return scores, movavg100, arrival_scores
