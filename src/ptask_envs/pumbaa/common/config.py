


class PumbaaTaskConfig():
    def __init__(self, task_cfg):
        self.task_cfg = task_cfg
        self.subtask_cfg = task_cfg['task']
        self.env_cfg = task_cfg['env']
        self.pumbaa_cfg = self.env_cfg['pumbaa']

        self.assets = task_cfg['asset']

        # task config
        self.state_space_dict = self.subtask_cfg.get('state_space')
        self.hidden_state_space_dict = self.subtask_cfg.get('hidden_state_space')
        self.max_episode_length = self.subtask_cfg.get('episodeLength')
        self.action_mode = self.subtask_cfg.get('actionMode', 'continuous_std_6')
        self.reward_coef = self.subtask_cfg.get('reward_coef')
        self.reset_info_maps_file = self.subtask_cfg.get('resetInfoMaps', 'data/reset_info_maps.yaml')
        self.reset_type = self.subtask_cfg.get('resetType', 'random')
        self.record_done = self.subtask_cfg.get('record_done', True)
        self.record_reward_item = self.subtask_cfg.get('record_reward_item', True)

        # pumbaa config
        self.max_v = self.pumbaa_cfg.get('max_v', 0.2)
        self.min_v = self.pumbaa_cfg.get('max_v', 0.2)
        self.max_w = self.pumbaa_cfg.get('max_w', 0.55)
        self.flipper_dt = self.pumbaa_cfg.get('flipper_dt', 4)
        self.flipper_pos_max = self.pumbaa_cfg.get('flipper_pos_max', 60)
        self.flipper_joint_max_vel = self.pumbaa_cfg.get('flipper_joint_max_vel', 20)
        self.flipper_joint_stiffness = self.pumbaa_cfg.get('flipper_joint_stiffness', 500)
        self.flipper_material_friction = self.pumbaa_cfg.get('flipper_material_friction', 1)
        self.wheel_material_friction = self.pumbaa_cfg.get('wheel_material_friction', 1)

        # env config
        self.is_follow_camera = self.env_cfg.get('follow_camera', False)
