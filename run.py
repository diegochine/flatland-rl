import numpy as np
import time

from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool

import keras.layers as layers
from keras.models import Model
from keras.optimizers import Adam
from rl.memory import SequentialMemory

import config as cfg
import logger as log
from processor import FlatLandProcessor
from multidqnagent import MultiDQNAgent

env = RailEnv(width=cfg.WIDTH, height=cfg.HEIGHT,
              rail_generator=sparse_rail_generator(),
              number_of_agents=cfg.NUMBER_OF_AGENTS,
              obs_builder_object=GlobalObsForRailEnv())
env_renderer = RenderTool(env)
logger = log.setup_logger('run', 'logs/run.txt')

inputs = layers.Input(shape=(env.width, env.height, 23))
x = layers.Flatten()(inputs)
x = layers.Dense(128, activation='relu')(x)
output = layers.Dense(cfg.NUM_ACTIONS, activation='linear')(x)
model = Model(inputs=inputs, outputs=output)
processor = FlatLandProcessor()
memory = SequentialMemory(limit=10000, window_length=1)
agent = MultiDQNAgent(model, memory=memory, processor=processor,
                      nb_agents=cfg.NUMBER_OF_AGENTS,
                      nb_actions=cfg.NUM_ACTIONS)
agent.compile(Adam(lr=1e-3), metrics=['mse'])
agent.fit(env, nb_episodes=cfg.N_EPISODES)
