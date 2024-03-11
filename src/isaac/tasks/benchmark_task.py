
import torch

from .crossing_task import CrossingTask


class BenchmarkTask(CrossingTask):
    def __init__(
            self,
            name,
            sim_config,
            env,
            offset=None
    ) -> None:
        CrossingTask.__init__(self, name, sim_config, env, offset)

    def get_states(self):
        return self.obs_buf

    def get_extras(self):
        return dict()

    def cleanup(self) -> None:
        super().cleanup()
        self.extras = dict()
