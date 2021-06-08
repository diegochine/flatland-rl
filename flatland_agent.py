from typing import Optional

import numpy as np

from agents import DQNAgent
from memory import Buffer
from networks import QNetwork
import tensorflow as tf

from utils import types


class FlatDqnAgent(DQNAgent):
    def __init__(self,
                 state_shape: tuple,
                 action_shape: tuple,
                 q_network: QNetwork,
                 optimizer: tf.keras.optimizers.Optimizer,
                 gamma: types.Float = 0.5,
                 epsilon: types.Float = 0.1,
                 epsilon_decay: types.Float = 0.98,
                 epsilon_min: types.Float = 0.01,
                 target_update_period: int = 500,
                 tau: types.Float = 1.0,
                 ddqn: bool = True,
                 buffer: Optional[Buffer] = None,
                 name: str = 'FlatlandDQNAgent',
                 training: bool = True,
                 save_dir: str = './output'):
        super(FlatDqnAgent, self).__init__(state_shape,
                                           action_shape,
                                           q_network,
                                           optimizer,
                                           gamma,
                                           epsilon,
                                           epsilon_decay,
                                           epsilon_min,
                                           target_update_period,
                                           tau,
                                           ddqn,
                                           buffer,
                                           name,
                                           training,
                                           save_dir)

    def _minibatch_to_tf(self, minibatch):
        """ Given a list of experience tuples (s_t, a_t, r_t, s_t+1, done_t)
            returns list of 5 tensors batches """
        state_batch = tf.convert_to_tensor([sample[0].reshape(self.state_shape)
                                            for sample in minibatch])
        action_batch = tf.convert_to_tensor([sample[1] for sample in minibatch])
        reward_batch = tf.convert_to_tensor([sample[2] for sample in minibatch])
        # FIXME when done new_state is none
        new_state_batch = tf.convert_to_tensor([sample[3].reshape(self.state_shape)
                                                for sample in minibatch])
        done_batch = np.array([sample[4] for sample in minibatch])
        return [state_batch, action_batch, reward_batch, new_state_batch, done_batch]
