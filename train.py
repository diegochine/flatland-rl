import os
import gin
import numpy as np
import wandb
from argparse import ArgumentParser

from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.utils.rendertools import RenderTool
from tensorflow.python.keras.optimizer_v2.adam import Adam

from pyagents.memory import PrioritizedBuffer
from pyagents.networks import QNetwork
from rail_env import RailEnvWrapper
from processor import normalize_observation
from flatland_agent import FlatlandDQNAgent

MED_SIZE = {
    'env': {'width': 20, 'height': 20, 'random_seed': 42},
    'rail_gen': {'max_num_cities': 3, 'max_rails_between_cities': 10, 'max_rails_in_city': 2, 'seed': 42}
}
BIG_SIZE = {
    'env': {'width': 45, 'height': 45, 'random_seed': 42},
    'rail_gen': {'max_num_cities': 8, 'max_rails_between_cities': 5, 'max_rails_in_city': 3, 'seed': 42}
}


@gin.configurable
def flatland_train(params, n_agents, tree_depth, state_shape, action_shape, n_episodes=500, steps_to_train=4,
                   batch_size=128, learning_rate=0.0001, max_steps=250, min_memories=1000, output_dir="./output/"):
    rail_generator = sparse_rail_generator(**params['rail_gen'])
    env = RailEnvWrapper(**params['env'],
                         rail_generator=rail_generator,
                         number_of_agents=n_agents,
                         obs_builder_object=TreeObsForRailEnv(max_depth=tree_depth))
    # env_renderer = RenderTool(env, screen_height=1500, screen_width=1500)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    buffer = PrioritizedBuffer()
    q_net = QNetwork(state_shape, action_shape)
    optim = Adam(learning_rate=learning_rate)
    player = FlatlandDQNAgent(state_shape, action_shape, q_network=q_net, buffer=buffer,
                              optimizer=optim)
    wandb.config.update({'learning_rate': learning_rate,
                         'batch_size': batch_size,
                         'steps_to_train': steps_to_train,
                         'n_agents': n_agents,
                         **player.get_config(),
                         **q_net.get_config(),
                         **buffer.get_config()})

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

        while not done['__all__'] and step < max_steps and not all(info['deadlock'].values()):
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
        trains_deadlocked = sum(info['deadlock'].values())
        arrival_scores.append(trains_arrived / n_agents)
        this_episode_score = np.mean(scores[-100:])
        this_episode_arrival = np.mean(arrival_scores[-100:])
        movavg100.append(this_episode_score)
        eps_history.append(player.epsilon)
        wandb.log({'score': score,
                   "arrivals": trains_arrived / n_agents,
                   "deadlocks": trains_deadlocked / n_agents,
                   "epsilon": player.epsilon})
        if episode % 10 == 0:
            print(f'=========================================\n'
                  f'EPISODE: {episode:4d}/{n_episodes:4d}, SCORE: {movavg100[-1]:4.0f}\n'
                  f'ARRIVAL RATE: {this_episode_arrival * 100:3.0f}%, EPS: {player.epsilon:.2f}\n'
                  f'=========================================)')

        if (episode % 500) == 0:
            player.save(ver=(episode // 500))

    player.save(ver='final')
    return scores, movavg100, eps_history, arrival_scores


if __name__ == "__main__":
    parser = ArgumentParser(description="Script for training flatland agents")
    parser.add_argument('-c', '--config', nargs='+', help='path to gin config file(s) ', required=True)
    parser.add_argument('-a', '--agents', type=int, help='number of agents', required=True)
    parser.add_argument('--big', dest='big_map', help='use big map (always used when #agents > 2)',
                        action='store_true', default=False)
    parser.add_argument('-k', '--key', type=str, help='API key for WandB', required=True)
    args = parser.parse_args()
    params = BIG_SIZE if args.big_map or args.agents > 2 else MED_SIZE
    results = []
    arrivals = []
    wandb.login(key=args.key)
    for cfg_file in args.config:
        gin.parse_config_file(cfg_file)
        wandb.init(project='flatland')
        print(f'******************************************************\n'
              f'STARTING TRAINING FOR {cfg_file}\n'
              f'******************************************************\n')
        info = flatland_train(params=params, n_agents=args.agents)
        results.append((cfg_file, info[:-1]))
        arrivals.append((cfg_file, info[-1]))
