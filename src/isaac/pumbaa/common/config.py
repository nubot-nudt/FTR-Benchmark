
class PumbaaTaskConfig():
    def __init__(self, task_cfg):
        self.task_cfg = task_cfg
        self.subtask_cfg = task_cfg['task']
        self.env_cfg = task_cfg['env']

        self.assets = task_cfg['asset']

        # task config
        self.state_space_dict = self.subtask_cfg.get('state_space')
        self.hidden_state_space_dict = self.subtask_cfg.get('hidden_state_space')
        self.max_episode_length = self.subtask_cfg.get('episodeLength')
        self.action_mode = self.subtask_cfg.get('actionMode', 'continuous_std_6')
        self.reward_coef = self.subtask_cfg.get('reward_coef')
        self.reset_info_maps_file = self.subtask_cfg.get('resetInfoMaps', 'data/reset_info_maps.yaml')
        self.reset_type = self.subtask_cfg.get('resetType', 'random')

        # env config
        self.max_v = self.env_cfg.get('max_v', 0.55)
        self.max_w = self.env_cfg.get('max_w', 0.55)
        self.flipper_dt = self.env_cfg.get('flipper_dt', 4)
        self.is_follow_camera = self.env_cfg.get('follow_camera', False)