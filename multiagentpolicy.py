import numpy as np
from pyagents.policies import QPolicy


class MultiAgentPolicy(QPolicy):

    def __init__(self, state_shape, action_shape, q_network, n_agents):
        super().__init__(state_shape, action_shape, q_network)
        self.n_agents = n_agents

    def _act(self, obs):
        qvals = self._q_network(obs.reshape(1, *obs.shape)).reshape(self.n_agents, self._action_shape/self.n_agents)
        action_dict = {a: v for a, v in enumerate(np.argmax(qvals, axis=0))}
        return action_dict
