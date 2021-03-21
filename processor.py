from rl.core import Processor


class FlatLandProcessor(Processor):

    def process_observation(self, observation):
        print(observation)
        return observation
