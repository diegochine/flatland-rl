import os
import random
import gin
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.utils.rendertools import RenderTool
from pyagents.utils import plot_result
from tensorflow.python.keras.optimizer_v2.adam import Adam

from pyagents.memory import PrioritizedBuffer
from pyagents.networks import QNetwork
from processor import normalize_observation
from flatland_agent import FlatlandDQNAgent


@gin.configurable
def flatland_train(width, height, n_agents, tree_depth, state_shape, action_shape, n_episodes=500, steps_to_train=4,
                   batch_size=128, learning_rate=0.0001, max_steps=250, min_memories=20000, output_dir="./output/"):
    rail_generator = sparse_rail_generator(max_num_cities=n_agents, seed=random.randint(0, 10000),
                                           max_rails_between_cities=10, max_rails_in_city=2)
    env = RailEnv(width=width, height=height,
                  rail_generator=rail_generator,
                  number_of_agents=n_agents,
                  obs_builder_object=TreeObsForRailEnv(max_depth=tree_depth),
                  random_seed=random.randint(0, 10000))
    # env_renderer = RenderTool(env)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    buffer = PrioritizedBuffer()
    q_net = QNetwork(state_shape, action_shape)
    optim = Adam(learning_rate=learning_rate)
    player = FlatlandDQNAgent(state_shape, action_shape, q_network=q_net, buffer=buffer,
                              optimizer=optim)
    scores = []
    arrival_scores = []
    movavg100 = []
    eps_history = []
    player.memory_init(env, max_steps, min_memories, list(range(5)), processor=normalize_observation)

    # Empty dictionary for all agent action
    action_dict = dict()

    for episode in range(1, n_episodes + 1):

        next_obs, info = env.reset()
        # env_renderer.reset()
        state = {a: normalize_observation(next_obs[a], tree_depth) for a in range(env.get_num_agents())}
        score = 0
        step = 0
        done = {'__all__': False}

        while not done['__all__'] and step < max_steps:
            # Chose an action for each agent in the environment
            for a in range(env.get_num_agents()):
                if info['action_required'][a]:
                    action = player.act(state[a])
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
                if state[a] is not None:
                    player.remember(state[a], action_dict[a], all_rewards[a], next_state[a], done[a])
                score += all_rewards[a]
            state = next_state
            if step % steps_to_train == 0:
                player.train(batch_size)
            step += 1

        scores.append(score)
        trains_arrived = sum([done[a] for a in range(env.get_num_agents())])
        arrival_scores.append(trains_arrived / n_agents)
        this_episode_score = np.mean(scores[-100:])
        this_episode_arrival = np.mean(arrival_scores[-100:])
        movavg100.append(this_episode_score)
        eps_history.append(player.epsilon)
        if episode % 10 == 0:
            print(f'===============================================\n'
                  f'EPISODE: {episode:4d}/{n_episodes:4d}, SCORE: {movavg100[-1]:4.0f}\n'
                  f'ARRIVAL RATE: {this_episode_arrival*100:3.0f}%, EPS: {player.epsilon:.2f}\n'
                  f'===============================================')

        if (episode % 500) == 0:
            player.save(ver=episode // 500)

    player.save(ver='final')
    return scores, movavg100, eps_history, arrival_scores


if __name__ == "__main__":
    parser = ArgumentParser(description="Script for training flatland agents")
    parser.add_argument('-c', '--config', nargs='+', help='path to gin config file(s) ', required=True)
    args = parser.parse_args()
    results = []
    arrivals = []
    for cfg_file in args.config:
        gin.parse_config_file(cfg_file)
        print(f'******************************************************\n'
              f'STARTING TRAINING FOR {cfg_file}\n'
              f'******************************************************\n')
        info = flatland_train()
        results.append((cfg_file, info[:-1]))
        arrivals.append((cfg_file, info[-1]))

    ncols = 2
    nrows = len(args.config) // 2
    plot_result((ncols, nrows), results)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex='all', sharey='all', constrained_layout=True)
    axes = axes.reshape(-1)
    for i, (name, arrival) in enumerate(arrivals):
        ax = axes[i]
        ax.set_title(name)
        ax.hist(arrival, bins=5, density=True)
        ax.set_ylabel('Bins size')
        ax.set_xlabel('Episode')
    plt.show()
