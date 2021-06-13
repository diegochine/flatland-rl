import os
from collections import deque

import gin
from argparse import ArgumentParser

import numpy as np
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import random_rail_generator
from flatland.utils.rendertools import RenderTool
from tensorflow.python.keras.optimizer_v2.adam import Adam

import processor
from memory import PrioritizedBuffer
from networks import QNetwork
from flatland_agent import FlatlandDQNAgent


@gin.configurable
def train_agent(width, height, n_agents, tree_depth, state_shape, action_shape, n_episodes=10001,
                batch_size=128, learning_rate=0.0001,  max_steps=500, min_memories=5000, output_dir="./output/"):
    env = RailEnv(width=width, height=height,
                  rail_generator=random_rail_generator(),
                  number_of_agents=n_agents,
                  obs_builder_object=TreeObsForRailEnv(max_depth=tree_depth))
    # env_renderer = RenderTool(env)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    buffer = PrioritizedBuffer()
    q_net = QNetwork(state_shape, action_shape)
    optim = Adam(learning_rate=learning_rate)
    player = FlatlandDQNAgent(state_shape, action_shape, q_network=q_net, buffer=buffer,
                              optimizer=optim)

    player.memory_init(env, max_steps, min_memories, list(range(5)), processor=processor.normalize_observation)

    # Empty dictionary for all agent action
    action_dict = dict()
    scores = deque(maxlen=100)

    for episode in range(n_episodes):

        next_obs, info = env.reset()
        state = {a: processor.normalize_observation(next_obs[a], tree_depth) for a in range(env.get_num_agents())}
        score = 0
        step = 0
        done = {'__all__': False}

        while not done['__all__'] and step < max_steps:
            # Chose an action for each agent in the environment
            for a in range(env.get_num_agents()):
                action = player.act(state[a])
                action_dict.update({a: action})

            next_obs, all_rewards, done, info = env.step(action_dict)
            # env_renderer.render_env(show=True, show_observations=True, show_predictions=False)

            next_state = {a: None for a in range(env.get_num_agents())}
            next_state.update({a: processor.normalize_observation(next_obs[a], tree_depth)
                               for a in range(env.get_num_agents())
                               if next_obs[a] is not None})

            for a in range(env.get_num_agents()):
                player.remember(state[a], action_dict[a], all_rewards[a], next_state[a], done[a])
                score += all_rewards[a]
            state = next_state
            if step % 100 == 0:
                player.train(batch_size)
            step += 1
        scores.append(score)
        print(f'EPISODE: {episode:4d}/{n_episodes:4d}, SCORE: {np.mean(scores):4.0f}, EPS: {player.epsilon}')

        if (episode % 1000) == 0:
            player.save(ver=episode//1000)

    player.save(ver='final')


if __name__ == "__main__":
    parser = ArgumentParser(description="Script for training flatland agents")
    parser.add_argument('-c', '--config', type=str, help='path to gin config file, ', required=True)
    args = parser.parse_args()
    gin.parse_config_file(args.config)
    train_agent()
