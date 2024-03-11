import torch

from .crossing_task import CrossingTask


class ExperimentTask(CrossingTask):
    def __init__(
            self,
            name,
            sim_config,
            env,
            offset=None
    ) -> None:

        CrossingTask.__init__(self, name, sim_config, env, offset)

    def _is_done_in_target(self, index):
        point = self.positions[index][:2]
        target = self.target_positions[index][:2]
        start = self.start_positions[index][:2]

        a = start - target
        b = point - target

        return torch.dot(a, b) <= 0

    # def post_physics_step(self):
    #     ret = super().post_physics_step()
    #     print(self.imus[0].get_lin_acc())
    #
    #     return ret

    def take_actions(self, actions, indices):

        ret = self._action_mode_execute.convert_actions_to_std_dict(actions, default_v=0.3, default_w=0)
        self.articulation_view.set_v_w(ret['vel'], indices=indices)
        self._flipper_control.set_pos_dt_with_max(ret['flipper'], 60, index=indices)
        self.articulation_view.set_all_flipper_position_targets(self._flipper_control.positions)

        return ret['vel'], ret['flipper']

    # def set_up_scene(self, scene) -> None:
    #     super().set_up_scene(scene, apply_imu=True)


