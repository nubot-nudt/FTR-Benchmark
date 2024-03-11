import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/common'))
from utils.path import apply_project_directory
apply_project_directory()

from experiments.base import ExperimentChain
from experiments.reward import ValueExperiment, RewardExperiment
from experiments.metrics import MetricsExperiment
from experiments.runner import create_runner


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