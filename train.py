import os
import numpy as np
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import random_rail_generator
from flatland.utils.rendertools import RenderTool
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop

import processor
import config as cfg
import logger as log
from drl.memory.prioritizedbuffer import PrioritizedBuffer
from drl.networks.qnetwork import QNetwork
from flatland_agent import FlatDqnAgent


def flatland_init(player, env, max_steps, min_memories, actions):
    while player.memory_len <= min_memories:
        next_obs, info = env.reset()
        state = {o: processor.process_observation(next_obs[o]) for o in next_obs}
        done = {'__all__': False}
        step = 0
        action_dict = dict()
        player._memory.commit_ltmemory()
        while not done['__all__'] and step < max_steps:
            for a in range(env.get_num_agents()):
                act = np.random.choice(actions, 1)[0]
                action_dict.update({a: act})
            next_obs, r, done, _ = env.step(action_dict)
            next_state = {o: processor.process_observation(next_obs[o]) for o in next_obs}
            for a in range(env.get_num_agents()):
                player.remember(state[a], action_dict[a], r[a], next_state[a], done[a])
            state = next_state
            step += 1

env = RailEnv(width=cfg.WIDTH, height=cfg.HEIGHT,
              rail_generator=random_rail_generator(),
              number_of_agents=cfg.NUMBER_OF_AGENTS,
              obs_builder_object=GlobalObsForRailEnv())
env_renderer = RenderTool(env)
logger = log.setup_logger('run', 'logs/run.txt')

if not os.path.exists(cfg.OUTPUT_DIR):
    os.makedirs(cfg.OUTPUT_DIR)

state_size = (env.width, env.height, 23)
action_size = cfg.NUM_ACTIONS
buffer = PrioritizedBuffer()
q_net = QNetwork(state_size, action_size, conv_layer_params=[(5, 3, 1), (3, 2, 1)], fc_layer_params=(256, 128))
player = FlatDqnAgent(state_size, action_size, q_network=q_net, buffer=buffer, epsilon=1, epsilon_decay=0.995,
                      optimizer=RMSprop(momentum=0.1), name='flatland_global')

logger.info("Memory Initialization")
flatland_init(player, env, cfg.MAX_STEPS, cfg.MIN_MEMORIES, list(range(5)))

# Empty dictionary for all agent action
action_dict = dict()
logger.info("Starting Training")

for episode in range(cfg.N_EPISODES):

    next_obs, info = env.reset()
    state = {o: processor.process_observation(next_obs[o]) for o in next_obs}
    score = 0
    step = 0
    done = {'__all__': False}

    while not done['__all__'] and step < cfg.MAX_STEPS:
        # Chose an action for each agent in the environment
        for a in range(env.get_num_agents()):
            action = player.act(state[a])
            action_dict.update({a: action})

        next_obs, all_rewards, done, info = env.step(action_dict)
        env_renderer.render_env(show=True, show_observations=True, show_predictions=False)

        next_state = {o: processor.process_observation(next_obs[o]) for o in next_obs}

        for a in range(env.get_num_agents()):
            player.remember(state[a], action_dict[a], all_rewards[a], next_state[a], done[a])
            score += all_rewards[a]
        state = next_state
        step += 1

    print(f'EPISODE: {episode:4d}/{cfg.N_EPISODES:4d}, SCORE: {score:4.0f}, EPS: {player.epsilon}')
    player.train(cfg.BATCH_SIZE)

    if (episode % 1000) == 0:
        player.save()

player.save()
