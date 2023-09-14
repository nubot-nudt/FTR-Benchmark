
import torch

from .common import PumbaaTask
class BenchmarkTask(PumbaaTask):
    def __init__(
            self,
            name,
            sim_config,
            env,
            offset=None
    ) -> None:
        self.max_v = 0.25
        PumbaaTask.__init__(self, name, sim_config, env, offset)
