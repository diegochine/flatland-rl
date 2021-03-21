from rl.core import Processor


class FlatLandProcessor(Processor):

    def process_observation(self, observation):
        obs = observation[0]
        for o in obs:
            pass
