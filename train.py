import os
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import random_rail_generator
from flatland.utils.rendertools import RenderTool
from keras import layers
from keras.models import Model
from keras.regularizers import l1_l2, l2

import processor
import config as cfg
import logger as log
from agent import DQNAgent

env = RailEnv(width=cfg.WIDTH, height=cfg.HEIGHT,
              rail_generator=random_rail_generator(),
              number_of_agents=cfg.NUMBER_OF_AGENTS,
              obs_builder_object=GlobalObsForRailEnv())
env_renderer = RenderTool(env)
logger = log.setup_logger('run', 'logs/run.txt')

if not os.path.exists(cfg.OUTPUT_DIR):
    os.makedirs(cfg.OUTPUT_DIR)

state_size = (env.width, env.height, 23)
inputs = layers.Input(shape=state_size)
x = layers.Flatten()(inputs)
x = layers.Dense(512, activation='relu',
                 kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                 bias_regularizer=l2(1e-4))(x)
x = layers.Dense(512, activation='relu',
                 kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                 bias_regularizer=l2(1e-4))(x)
x = layers.Dense(256, activation='relu',
                 kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                 bias_regularizer=l2(1e-4))(x)
x = layers.Dense(256, activation='relu',
                 kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                 bias_regularizer=l2(1e-4))(x)
x = layers.Dense(128, activation='relu',
                 kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                 bias_regularizer=l2(1e-4))(x)
output = layers.Dense(cfg.NUM_ACTIONS, activation='linear')(x)
q_network = Model(inputs=inputs, outputs=output)

agent = DQNAgent(state_size, cfg.NUM_ACTIONS, model=q_network)

# Empty dictionary for all agent action
action_dict = dict()
logger.info("Starting Training")

for episode in range(cfg.N_EPISODES):

    next_obs, info = env.reset()
    state = {o: processor.process_observation(next_obs[o]) for o in next_obs}
    score = 0
    step = 0
    done = {'__all__': False}

    while not done['__all__'] and step < 1000:
        # env.render()
        # Chose an action for each agent in the environment
        for a in range(env.get_num_agents()):
            action = agent.act(state[a])
            action_dict.update({a: action})

        next_obs, all_rewards, done, info = env.step(action_dict)
        # env_renderer.render_env(show=True, show_observations=True, show_predictions=False)

        next_state = {o: processor.process_observation(next_obs[o]) for o in next_obs}

        for a in range(env.get_num_agents()):
            agent.remember(state[a], action_dict[a], all_rewards[a], next_state[a], done[a])
            score += all_rewards[a]
        state = next_state

        if done:
            print(f'EPISODE: {episode:4d}/{cfg.N_EPISODES:4d}, SCORE: {score:4d}, EPS: {agent.epsilon}')
        step += 1

    agent.replay(cfg.BATCH_SIZE)

    if (episode % 1000) == 0:
        agent.save(cfg.OUTPUT_DIR + f"/ep{episode}.hdf5")

agent.save(cfg.OUTPUT_DIR + "endrun.hdf5")
