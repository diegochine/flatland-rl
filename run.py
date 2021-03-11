from symbol import import_from

import numpy as np
import time

from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_generators import random_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool

import config as cfg
import logger as log
from tensorflow.keras.layers as layers
from tensorflow.keras.model import Model
from tensorflow.keras.optimizers import Adam
from agent import Agent
from rl.agents import DQNAgent


env = RailEnv(width=20, height=20,
              rail_generator=random_rail_generator(),
              number_of_agents=cfg.NUMBER_OF_AGENTS,
              obs_builder_object=GlobalObsForRailEnv())
env_renderer = RenderTool(env)
logger = log.setup_logger('run', 'logs/run.txt')

#  agent = Agent()

input_block = layers.Flatten(input_shape=(20,20,23))
x = layers.Dense(128, activation='relu')(input_block)
output = layers.Dense(cfg.NUM_ACTIONS, activation'linear')(x)
model = Model(input=input_block, outputs=output)

# only model required
kerasAgent = DQNAgent(model)
kerasAgent.compile(Adam(lr=1e-3), metrics=['mse'])
kerasAgent.fit(env, nb_steps=10000, visualize=True, verbose=2)

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
