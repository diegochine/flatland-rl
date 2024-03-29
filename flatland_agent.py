from typing import Optional

import gin
import numpy as np
import tensorflow as tf

from pyagents.agents import DQNAgent
from pyagents.memory import Buffer
from pyagents.networks import QNetwork
from pyagents.utils import types


@gin.configurable
class FlatlandDQNAgent(DQNAgent):
    def __init__(self,
                 state_shape: tuple,
                 action_shape: tuple,
                 q_network: QNetwork,
                 optimizer: tf.keras.optimizers.Optimizer,
                 gamma: types.Float = 0.99,
                 epsilon: types.Float = 0.1,
                 epsilon_decay: types.Float = 0.99,
                 epsilon_min: types.Float = 0.01,
                 target_update_period: int = 100,
                 tau: types.Float = 1.0,
                 ddqn: bool = True,
                 buffer: Optional[Buffer] = None,
                 name: str = 'FlatlandDQNAgent',
                 training: bool = True,
                 save_dir: str = './output'):
        super(FlatlandDQNAgent, self).__init__(state_shape=state_shape,
                                               action_shape=action_shape,
                                               q_network=q_network,
                                               optimizer=optimizer,
                                               gamma=gamma,
                                               epsilon=epsilon,
                                               epsilon_decay=epsilon_decay,
                                               epsilon_min=epsilon_min,
                                               target_update_period=target_update_period,
                                               tau=tau,
                                               ddqn=ddqn,
                                               buffer=buffer,
                                               name=name,
                                               training=training,
                                               save_dir=save_dir)

    def memory_init(self, env, max_steps, min_memories, actions=None, processor=None):
        print('Initializing memories buffer')
        while self.memory_len <= min_memories:
            next_obs, info = env.reset()
            if processor is not None:
                state = processor.process(next_obs)
            else:
                state = next_obs
            done = {'__all__': False}
            step = 0
            action_dict = dict()
            self._memory.commit_ltmemory()
            while not done['__all__'] and step < max_steps:
                for a in range(env.get_num_agents()):
                    if info['action_required'][a]:
                        action = np.random.choice(actions, 1)[0]
                    else:
                        action = 0
                    action_dict.update({a: action})
                next_obs, r, done, info = env.step(action_dict)
                if processor is not None:
                    next_state = processor.process(next_obs)
                else:
                    next_state = next_obs
                for a in range(env.get_num_agents()):
                    if state[a] is not None:
                        self.remember(state[a], action_dict[a], r[a], next_state[a], done[a])
                state = next_state
                step += 1

    def remember(self, state, action, reward, next_state, done):
        """
        Saves piece of memory
        :param state: state at current timestep
        :param action: action at current timestep
        :param reward: reward at current timestep
        :param next_state: state at next timestep
        :param done: whether the episode has ended
        :return:
        """
        if done:
            next_state = np.zeros_like(state)
        self._memory.commit_stmemory([state, action, reward, next_state, done])

    def _minibatch_to_tf(self, minibatch):
        """ Given a list of experience tuples (s_t, a_t, r_t, s_t+1, done_t)
            returns list of 5 tensors batches """
        state_batch = tf.convert_to_tensor([sample[0].reshape(self.state_shape)
                                            for sample in minibatch])
        action_batch = tf.convert_to_tensor([sample[1] for sample in minibatch])
        reward_batch = tf.convert_to_tensor([sample[2] for sample in minibatch])
        new_state_batch = tf.convert_to_tensor([sample[3].reshape(self.state_shape)
                                                for sample in minibatch])
        done_batch = np.array([sample[4] for sample in minibatch])
        return [state_batch, action_batch, reward_batch, new_state_batch, done_batch]
