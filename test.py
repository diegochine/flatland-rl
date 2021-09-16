import numpy as np
import matplotlib.pyplot as plt
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.utils.rendertools import RenderTool

from flatland_agent import FlatlandDQNAgent
from processor import normalize_observation
from rail_env import RailEnvWrapper


def flatland_test(path, width, height, n_agents, tree_depth, n_episodes=10, max_steps=200):
    rail_generator = sparse_rail_generator(max_num_cities=n_agents,
                                           max_rails_between_cities=10, max_rails_in_city=2,
                                           seed=42)
    tree_obs = TreeObsForRailEnv(max_depth=tree_depth, predictor=ShortestPathPredictorForRailEnv())
    env = RailEnvWrapper(width=width, height=height,
                         rail_generator=rail_generator,
                         number_of_agents=n_agents,
                         obs_builder_object=tree_obs,
                         random_seed=42)
    env_renderer = RenderTool(env)

    player = FlatlandDQNAgent.load(path, 'final', training=False)

    scores = []
    movavg100 = []
    arrival_scores = []

    # Empty dictionary for all agent action
    action_dict = dict()

    for episode in range(n_episodes):

        next_obs, info = env.reset()
        env_renderer.reset()
        state = {a: normalize_observation(next_obs[a], tree_depth) for a in range(env.get_num_agents())}
        score = 0
        step = 0
        done = {'__all__': False}

        while not done['__all__'] and step < max_steps and not all(info['deadlock'].values()):
            # Chose an action for each agent in the environment
            for a in range(env.get_num_agents()):
                if info['action_required'][a]:
                    action = player.act(state[a])
                else:
                    action = 0
                action_dict.update({a: action})

            next_obs, all_rewards, done, info = env.step(action_dict)

            env_renderer.render_env(show=True, show_observations=True, show_predictions=False)

            next_state = {a: None for a in range(env.get_num_agents())}
            next_state.update({a: normalize_observation(next_obs[a], tree_depth)
                               for a in range(env.get_num_agents())
                               if next_obs[a] is not None})

            for a in range(env.get_num_agents()):
                player.remember(state[a], action_dict[a], all_rewards[a], next_state[a], done[a])
                score += all_rewards[a]
            state = next_state
            step += 1

        trains_arrived = sum([done[a] for a in range(n_agents)])
        arrival_scores.append(trains_arrived / n_agents)
        scores.append(score)
        this_episode_score = np.mean(scores[-100:])
        movavg100.append(this_episode_score)
        print(
            f'EPISODE: {episode:4d}/{n_episodes:4d}, SCORE: {movavg100[-1]:4.0f}, ARRIVAL RATE: {arrival_scores[-1] * 100:3.0f}%')

    env_renderer.close_window()
    return scores, movavg100, arrival_scores


if __name__ == "__main__":
    arrivals = []
    n_agents = 4
    for dir in ('output/flatland_tree1', 'output/flatland_tree2',
                'output/flatland_tree3', 'output/flatland_tree4'):
        info = flatland_test(dir, 20, 20, n_agents, 2)
        scores, moveavg, arrival_scores = info
        arrivals.append(arrival_scores)

    fig, axes = plt.subplots(nrows=2, ncols=2, sharex='all', sharey='all', constrained_layout=True)
    axes = axes.reshape(-1)
    for i, arrival in enumerate(arrivals):
        ax = axes[i]
        ax.set_title(f'cfg {i}')
        weights = np.ones_like(arrival) / len(arrival)
        _, bins, _ = ax.hist(arrival, bins=n_agents + 1, weights=weights)
        ax.set_ylabel('Bins size')
        ax.set_xlabel('Episode')
        ax.set_xticks(bins)
        ax.set_xticklabels([f'{i} trains' for i in range(len(bins))])
    plt.show()
