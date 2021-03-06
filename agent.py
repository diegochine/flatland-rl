import numpy as np

import config as cfg
import logger as log


class Agent:

    def __init__(self, name='agent'):
        self._logger = log.setup_logger(name, f'{"logs/" + name + ".txt"}')

    def act(self, state):
        """
        :param state: observation
        :return action given current policy
        """
        return np.random.randint(0, 5)

    def learn(self, memories):
        """
        :param memories:
        """
        pass
