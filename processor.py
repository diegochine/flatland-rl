import numpy as np
from rl.core import Processor


class FlatLandProcessor(Processor):

    def process_observation(self, observation):
        return np.concatenate(observation, axis=2)
