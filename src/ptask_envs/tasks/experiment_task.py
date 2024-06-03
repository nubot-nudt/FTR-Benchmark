import torch

from .crossing_task import CrossingTask
from .utils.geo import distance_to_line_2d, point_in_rotated_ellipse


class ExperimentTask(CrossingTask):
    def __init__(
            self,
            name,
            sim_config,
            env,
            offset=None
    ) -> None:

        CrossingTask.__init__(self, name, sim_config, env, offset)
        self.fixed_speed = self.cfg['task']['experiment']['fixed_speed']

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if self.fixed_speed >= 0:
            self.default_v[env_ids] = self.fixed_speed

    def _is_done_in_target(self, index):
        point = self.positions[index][:2]
        target = self.target_positions[index][:2]
        start = self.start_positions[index][:2]

        a = start - target
        b = point - target

        return torch.dot(a, b) <= 0

    def _is_done_out_of_range(self, index):
        point = self.positions[index]
        center = (self.start_positions[index] + self.target_positions[index]) / 2

        op = self.target_positions[index] - self.start_positions[index]
        d_max = op[:2].norm()
        theta = torch.arctan(op[1] / op[0])

        return not point_in_rotated_ellipse(
            point[0], point[1],
            center[0], center[1],
            d_max / 2 + 0.5, d_max / 4 + 0.5,
            theta
        )

    def register_gym_func(self):
        super().register_gym_func()
        self.done_components.update({
            'target': self._is_done_in_target,
            'out_of_range': self._is_done_out_of_range,
            'rollover': lambda i: torch.any(torch.abs(torch.rad2deg(self.orientations_3[i][:2])) >= 90),
            'timeout': lambda i: self.num_steps[i] >= self.max_step,
            'deviation': lambda i: 0.5 - distance_to_line_2d(self.start_positions[i], self.target_positions[i],
                                                             self.positions[i]) < 0,
            'overspeed': self._is_done_overspeed,
        })





