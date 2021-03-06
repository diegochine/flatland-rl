import numpy as np
import time

from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.rail_generators import random_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool

import config as cfg
import logger as log
from agent import Agent

transition_probability = [1.0,  # empty cell - Case 0
                          1.0,  # Case 1 - straight
                          1.0,  # Case 2 - simple switch
                          0.3,  # Case 3 - diamond drossing
                          0.5,  # Case 4 - single slip
                          0.5,  # Case 5 - double slip
                          0.2,  # Case 6 - symmetrical
                          0.0,  # Case 7 - dead end
                          0.2,  # Case 8 - turn left
                          0.2,  # Case 9 - turn right
                          1.0]  # Case 10 - mirrored switch
env = RailEnv(width=20, height=20,
              rail_generator=random_rail_generator(cell_type_relative_proportion=transition_probability),
              number_of_agents=cfg.NUMBER_OF_AGENTS,
              obs_builder_object=TreeObsForRailEnv(max_depth=2))
env_renderer = RenderTool(env)
logger = log.setup_logger('run', 'logs/run.txt')

agent = Agent()

# Empty dictionary for all agent action
action_dict = dict()
logger.info("Starting Training")

for episode in range(cfg.N_EPISODES):
    logger.info(f'EPISODE {episode:2d}/{cfg.N_EPISODES:2d}')
    # Reset environment and get initial observations for all agents
    obs, info = env.reset()
    for idx in range(env.get_num_agents()):
        tmp_agent = env.agents[idx]
        tmp_agent.speed_data["speed"] = 1 / (idx + 1)
    env_renderer.reset()
    # Here you can also further enhance the provided observation by means of normalization
    # See training navigation example in the baseline repository

    score = 0
    step = 0
    done = {'__all__': False}
    # Run episode
    while not done['__all__'] and step < cfg.MAX_STEPS:
        logger.info(f'STEP {step:3d}/{cfg.MAX_STEPS:3d}')
        # Chose an action for each agent in the environment
        for a in range(env.get_num_agents()):
            action = agent.act(obs[a])
            action_dict.update({a: action})
        # Environment step which returns the observations for all agents, their corresponding
        # reward and whether their are done
        next_obs, all_rewards, done, _ = env.step(action_dict)
        env_renderer.render_env(show=True, show_observations=True, show_predictions=False)

        # Update replay buffer and train agent
        for a in range(env.get_num_agents()):
            agent.learn((obs[a], action_dict[a], all_rewards[a], next_obs[a], done[a]))
            score += all_rewards[a]
        obs = next_obs.copy()
        step += 1
    logger.info(f'FINAL SCORE: {score:3.0f}')
