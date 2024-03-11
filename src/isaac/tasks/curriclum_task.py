
from itertools import cycle

import torch
from pipetools import pipe, X

from loguru import logger
from .crossing_task import CrossingTask


class CurriculumTask(CrossingTask):
    def __init__(
            self,
            name,
            sim_config,
            env,
            offset=None
    ) -> None:

        self._curriculum_config = sim_config.task_config['task']['curriculum']
        self._level_max = self._curriculum_config['level']
        self._level_num = 0
        self._last_update_step = 0
        self._update_step = self._curriculum_config['update_step']
        self._sort_by = {
            'point_x': lambda x: abs(max(x['start_point'][0], x['target_point'][0]))
        }[self._curriculum_config['sort_by']]

        CrossingTask.__init__(self, name, sim_config, env, offset)

    def _curriculum_reset_info_generate(self):
        if not hasattr(self, '_curriculum_reset_func'):
            self._curriculum_reset_func = lambda: next(self._curriculum_reset_data)

        return self._curriculum_reset_func()

    def prepare_reset_info(self):
        super().prepare_reset_info()

        self.__data = [self._sort_by(i) for i in self._reset_info]
        self.__min = min(self.__data)
        self.__max = max(self.__data)
        self.__one = (self.__max - self.__min) / self._level_max

        self._reset_info_generate = self._curriculum_reset_info_generate

    def _reset_idx_robot_info(self, num_resets):
        # print(self.physics_step_num)
        if self.physics_step_num >= self._last_update_step and self._level_num < self._level_max:
            self._level_num += 1
            self._last_update_step += self._update_step
            threshold = self.__min + self.__one * self._level_num
            datas = [i for i, j in zip(self._reset_info, self.__data) if j <= threshold + 1e-3]
            self._curriculum_reset_data = cycle(datas)
            logger.info(f'current curriculum level is {self._level_num}/{self._level_max}, '
                        f'and reset info num is {len(datas)}/{len(self._reset_info)}. '
                        f'step={self.physics_step_num}, {threshold=}.')

        return super()._reset_idx_robot_info(num_resets)



