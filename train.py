import os
import gin
import numpy as np
import wandb
from argparse import ArgumentParser

from flatland.envs.rail_env import RailEnvActions
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.utils.rendertools import RenderTool

from tensorflow.python.keras.optimizer_v2.adam import Adam

from pyagents.memory import PrioritizedBuffer, UniformBuffer
from pyagents.networks import QNetwork
from rail_env import RailEnvWrapper
from processor import normalize_observation
from flatland_agent import FlatlandDQNAgent
from eval import flatland_test

MED_SIZE = {
    'env': {'width': 48, 'height': 27, 'random_seed': 42},
    'rail_gen': {'max_num_cities': 5, 'max_rails_between_cities': 2, 'max_rails_in_city': 3, 'seed': 42}
}
BIG_SIZE = {
    'env': {'width': 64, 'height': 36, 'random_seed': 42},
    'rail_gen': {'max_num_cities': 9, 'max_rails_between_cities': 5, 'max_rails_in_city': 5, 'seed': 42}
}


@gin.configurable
def flatland_train(params, n_agents, tree_depth, state_shape, action_shape, n_episodes=1000, episodes_to_test=50,
                   steps_to_train=8, batch_size=128, learning_rate=0.0001, max_steps=250, min_memories=10000,
                   iterative=True, warm_up_episodes=200, use_wandb=False, prioritized=True, output_dir="./output/"):
    buffer = PrioritizedBuffer() if prioritized else UniformBuffer()
    q_net = QNetwork(state_shape, action_shape)
    optim = Adam(learning_rate=learning_rate)
    player = FlatlandDQNAgent(state_shape, action_shape, q_network=q_net, buffer=buffer, optimizer=optim)
    if use_wandb:
        wandb.config.update({'learning_rate': learning_rate,
                             'batch_size': batch_size,
                             'steps_to_train': steps_to_train,
                             'n_agents': n_agents,
                             **player.get_config(),
                             **q_net.get_config(),
                             **buffer.get_config()})

    rail_generator = sparse_rail_generator(**params['rail_gen'])
    tree_obs = TreeObsForRailEnv(max_depth=tree_depth, predictor=ShortestPathPredictorForRailEnv())

    if iterative:
        print(f'=========================================\n'
              f'STARTING WARM UP\n'
              f'=========================================')
        eps_decay = player.policy.get('_epsilon_decay')
        eps = player.policy.get('_epsilon')
        player.policy.set('_epsilon', player.policy.get('_epsilon_min'))
        envs = [RailEnvWrapper(**params['env'],
                               rail_generator=rail_generator,
                               number_of_agents=n,
                               obs_builder_object=tree_obs) for n in range(1, n_agents)]
        for e in envs:
            player.memory_init(e, max_steps, min_memories//(n_agents-1), list(range(5)), processor=normalize_observation)

        for episode in range(1, warm_up_episodes + 1):
            done, info, score = run_episode(action_shape, batch_size, envs[episode % (n_agents-1)], max_steps, n_agents,
                                            player, steps_to_train, tree_depth)
        player.policy.set('_epsilon', eps)
        player.policy.set('_epsilon_decay', eps_decay)
        print(f'=========================================\n'
              f'END OF WARM UP\n'
              f'=========================================')

    env = RailEnvWrapper(**params['env'],
                         rail_generator=rail_generator,
                         number_of_agents=n_agents,
                         obs_builder_object=tree_obs)
    # env_renderer = RenderTool(env, screen_height=1500, screen_width=1500)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    scores = []
    arrival_scores = []
    movavg100 = []
    player.memory_init(env, max_steps, min_memories, list(range(5)), processor=normalize_observation)

    for episode in range(1, n_episodes + 1):

        done, info, score = run_episode(action_shape, batch_size, env, max_steps, n_agents, player, steps_to_train,
                                        tree_depth)
        scores.append(score)
        trains_arrived = sum([done[a] for a in range(env.get_num_agents())])
        trains_deadlocked = sum(info['deadlock'].values())
        arrival_scores.append(trains_arrived / n_agents)
        this_episode_score = np.mean(scores[-100:])
        this_episode_arrival = np.mean(arrival_scores[-100:])
        movavg100.append(this_episode_score)
        if use_wandb:
            wandb.log({'train': {'score': score,
                                 "arrivals": trains_arrived,
                                 "deadlocks": trains_deadlocked,
                                 "epsilon": player.epsilon}})
        if episode % 10 == 0:
            print(f'=========================================\n'
                  f'EPISODE: {episode:4d}/{n_episodes:4d}, SCORE: {movavg100[-1]:4.0f}\n'
                  f'ARRIVAL RATE: {this_episode_arrival * 100:3.0f}%, EPS: {player.epsilon:.2f}\n'
                  f'=========================================)')

        if episode % episodes_to_test == 0:
            player.toggle_training()
            flatland_test(params['env']['width'], params['env']['height'], action_shape, n_agents, tree_depth,
                          max_num_cities=params['rail_gen']['max_num_cities'],
                          max_rails_between_cities=params['rail_gen']['max_rails_between_cities'],
                          max_rails_in_city=params['rail_gen']['max_rails_in_city'],
                          agent=player, n_episodes=20, max_steps=max_steps, use_wandb=use_wandb)
            player.save(ver=(episode // episodes_to_test))
            player.toggle_training()

    player.save(ver='final')
    return scores, movavg100, arrival_scores


def run_episode(action_shape, batch_size, env, max_steps, n_agents, player, steps_to_train, tree_depth):
    # Empty dictionary for all agent action
    action_dict = dict()
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
            if state[a] is not None:
                player.remember(state[a], action_dict[a], all_rewards[a], next_state[a], done[a])
            score += all_rewards[a] / n_agents
        state = next_state
        if step % steps_to_train == 0:
            player.train(batch_size)
        step += 1
    return done, info, score


if __name__ == "__main__":
    parser = ArgumentParser(description="Script for training flatland agents")
    parser.add_argument('-c', '--config', nargs='+', help='path to gin config file(s) ', required=True)
    parser.add_argument('-a', '--agents', type=int, help='number of agents', required=True)
    parser.add_argument('--big', dest='big_map', help='use big map (always used when #agents > 2)',
                        action='store_true', default=False)
    parser.add_argument('-k', '--key', type=str, help='API key for WandB (leave empty to avoid using it)', default='')
    parser.add_argument('--i', dest='iterative', help='Enable iterative multi agent training',
                        action='store_true', default=False)
    args = parser.parse_args()
    params = BIG_SIZE if args.big_map or args.agents > 3 else MED_SIZE
    use_wandb = bool(args.key)
    if use_wandb:
        wandb.login(key=args.key)
    for cfg_file in args.config:
        gin.parse_config_file(cfg_file)
        print(f'******************************************************\n'
              f'STARTING TRAINING FOR {cfg_file}\n'
              f'******************************************************\n')
        if use_wandb:
            run = wandb.init(project='flatland', entity='mazelions',
                             group=f"{'BIG_SIZE' if args.big_map or args.agents > 3 else 'MED_SIZE'}", reinit=True)
            info = flatland_train(params=params, n_agents=args.agents, iterative=args.iterative, use_wandb=use_wandb)
            run.finish()
        else:
            info = flatland_train(params=params, n_agents=args.agents, iterative=args.iterative, use_wandb=use_wandb)
