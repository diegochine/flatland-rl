import numpy as np


def process_observation(observation):
    if observation is None:
        return None
    return np.concatenate(observation, axis=2)
