
from .crossing_task import CrossingTask

class ManualTask(CrossingTask):
    def __init__(
            self,
            name,
            sim_config,
            env,
            offset=None
    ) -> None:

        CrossingTask.__init__(self, name, sim_config, env, offset)




