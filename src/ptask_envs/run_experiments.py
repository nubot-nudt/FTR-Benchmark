
from ptask_envs.experiments.base import ExperimentChain
from ptask_envs.experiments.reward import RewardExperiment
from ptask_envs.experiments.metrics import MetricsExperiment
from ptask_envs.experiments.runner import create_runner


if __name__ == '__main__':
    create_runner(
        ExperimentChain(
            [
                # ValueExperiment(),
                RewardExperiment(),
                MetricsExperiment()]
        ),
        is_http_pub=True,
    )