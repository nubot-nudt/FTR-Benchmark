class Experiment:

    def __init__(self):
        pass

    def before_init(self, cfg_dict, env):
        self.cfg = cfg_dict
        self.env = env
        self.task = env.task

    def after_init(self, runner):
        self.runner = runner
        self.executor = runner.executor

    def on_start(self):
        pass

    def on_end(self):
        pass

    def before_episode(self, epoch_num):
        self.epoch_num = epoch_num

    def after_episode(self, epoch_num):
        pass

    def before_step(self, step_num, obs, action):
        self.step_num = step_num

    def after_step(self, step_num, obs, reward, done, info):
        pass


class ExperimentChain(Experiment):

    def __init__(self, experiments):
        super().__init__()
        self.experiments = experiments

    def before_init(self, cfg_dict, env):
        super().before_init(cfg_dict, env)
        for e in self.experiments:
            e.before_init(cfg_dict, env)

    def after_init(self, runner):
        super().after_init(runner)
        for e in self.experiments:
            e.after_init(runner)

    def on_start(self):
        super().on_start()
        for e in self.experiments:
            e.on_start()

    def on_end(self):
        super().on_end()
        for e in self.experiments:
            e.on_end()

    def before_episode(self, epoch_num):
        super().before_episode(epoch_num)
        for e in self.experiments:
            e.before_episode(epoch_num)

    def after_episode(self, epoch_num):
        super().after_episode(epoch_num)
        for e in self.experiments:
            e.after_episode(epoch_num)

    def before_step(self, step_num, obs, action):
        super().before_step(step_num, obs, action)
        for e in self.experiments:
            e.before_step(step_num, obs, action)

    def after_step(self, step_num, obs, reward, done, info):
        super().after_step(step_num, obs, reward, done, info)
        for e in self.experiments:
            e.after_step(step_num, obs, reward, done, info)