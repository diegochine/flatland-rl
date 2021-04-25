import numpy as np
from random import sample
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.regularizers import l1_l2, l2
import memory as mem
import config as cfg


class DQNAgent:

    def __init__(self, state_shape, action_shape, model=None, gamma=cfg.GAMMA, learning_rate=cfg.LEARNING_RATE,
                 epsilon=cfg.EPSILON, epsilon_decay=cfg.EPSILON_DECAY, epsilon_min=cfg.EPSILON_MIN,
                 memory_size=cfg.MEMORY_SIZE, min_memories=cfg.MIN_MEMORIES):
        self.state_shape = state_shape
        self.action_shape = action_shape

        self.memory = mem.Memory(size_long=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.min_memories = min_memories

        if model is None:
            self.model = self._build_model()
        else:
            self.model = model
            self.model.compile(loss='mse', optimizer=Adam(self.learning_rate))

    def _build_model(self):
        model = Sequential()

        model.add(Dense(64, activation='relu', input_dim=self.state_shape,
                        kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                        bias_regularizer=l2(1e-4)))
        model.add(Dense(32, activation='relu',
                        kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                        bias_regularizer=l2(1e-4)))
        model.add(Dense(16, activation='relu',
                        kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                        bias_regularizer=l2(1e-4)))
        # output layer, linear because the output is
        model.add(Dense(self.action_shape, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(self.learning_rate))

        return model

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
        self.memory.commit_stmemory({'state': state, 'action': action, 'reward': reward,
                                     'next_state': next_state, 'q_value': None, 'done': done})

    def act(self, state):
        """
        :param state: current state
        :return: the best action according to current policy
        """
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_shape)
        qvals = self.model.predict(state.reshape(1, *state.shape))
        return np.argmax(qvals)

    def replay(self, batch_size):
        """
        Performs experience replay
        :param batch_size:
        :return:
        """
        q_value = -50
        for fragment in reversed(self.memory.last_episode()):
            if fragment['done']:
                q_value = fragment['reward']
                fragment['q_value'] = q_value
            else:
                q_value += self.gamma * fragment['reward']
                fragment['q_value'] = q_value
            print(q_value)

        self.memory.commit_ltmemory()

        if len(self.memory) > self.min_memories:
            minibatch = self.memory.sample(batch_size)

            for fragment in minibatch:
                target = fragment['q_value']
                action = fragment['action']
                state = fragment['state']
                state = state.reshape(1, *state.shape)
                target_f = self.model.predict(state)
                target_f[0][action] = target

                self.model.fit(state, target_f, epochs=1, verbose=0)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save(name)
