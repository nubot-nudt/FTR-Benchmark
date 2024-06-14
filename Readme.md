
# FTR-Bench: Benchmarking Deep Reinforcement Learning for Flipper-Track Robot Control

## Introduce

This project uses the NuBot rescue robot as the platform and obstacles crossing as the task. It builds a reinforcement learning training system in Isaac Sim and implements commonly used reinforcement learning algorithms.

The core objective of FTR-Bench is to establish a learning framework that enables the articulated tracked robot to navigate obstacles in various terrains. Specifically, FTR-Bench consists of three primary components: the simulation, tasks, and learning algorithms.

![](./docs/images/framework.png)

The flipper tracked robots have the capability to learn in various terrains. Considering real-world rescue robot applications, we have created four types of terrains, including flat ground, obstacles, bollards, and stairs. The task defines the conditions and objectives that the tracked robots need to meet in each scenario.

Eventually, our experiments demonstrate that RL can facilitate the robots to achieve some remarkable performance on such challenging tasks, enabling the agent to learn the optimal obstacle-navigation strategies within the environment. And there is still some room for improvement and more difficult tasks for future work.

### Tasks Design
|                               Task Name                               |                                                                                              Description                                                                                              |                           Demo                            |
|:---------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------:|
|            [Up the Steps](cfg/task/benchmark/StepsUp.yaml)            |                                                   Simulate the tracked robot ascending steps. The height of the steps ranges from 0.2$m$ to 0.4$m$.                                                   |  <img src="docs/images/tasks/stepsup.png" width="250"/>   |
|          [Down the Steps](cfg/task/benchmark/StepsDown.yaml)          |                                                  Simulate the tracked robot descending steps. The height of the steps ranges from 0.2$m$ to 0.6$m$.                                                   | <img src="docs/images/tasks/stepsdown.png" width="250"/>  |
|          [Cross the Uplifts](cfg/task/benchmark/Uplift.yaml)          |                         Simulate the tracked robot crossing over uplifts. The height of the uplifts ranges from 0.2$m$ to 0.4$m$, and the width ranges from 0.2$m$ to 0.5$m$.                         |  <img src="docs/images/tasks/uplifts.png" width="250"/>   |
| [Through Single-sided Steps](cfg/task/benchmark/UnilateralSteps.yaml) |                                      Simulate a tracked robot going through a single-side step. The height of the single-side step ranges from 0.1$m$ to 0.3$m$.                                      |  <img src="docs/images/tasks/unisteps.png" width="250"/>  |
|       [Drive on Plum Piles](cfg/task/benchmark/PlumPiles.yaml)        |               Simulate the tracked vehicle navigating on plum piles. The size of each individual plum pile ranges from 0.2$m$ to 0.5$m$, and the height ranges from 0.2$m$ to 0.5$m$.                 | <img src="docs/images/tasks/plumpiles.png" width="250"/>  |
|        [Drive on Wave Terrains](cfg/task/benchmark/Waves.yaml)        | Simulate the tracked vehicle navigating through waveform terrains. The size of each individual peak in the waveform terrain ranges from 0.25$m$ to 1$m$, and the height ranges from 0.1$m$ to 0.4$m$. |   <img src="docs/images/tasks/waves.png" width="250"/>    |
|           [Cross the Rails](cfg/task/benchmark/Rails.yaml)            |                 Simulate the tracked robot crossing over rails. The height of the railing ranges from 0.2$m$ to 0.3$m$, and the angle with the vehicle ranges from 10 to 45 degrees.                  |   <img src="docs/images/tasks/rails.png" width="250"/>    |
|           [Up the Stairs](cfg/task/benchmark/StairsUp.yaml)           |              Simulate the tracked vehicle climbing stairs. The width of each step of the stairs ranges from 0.2$m$ to 0.3$m$, and the slope of the stairs ranges from 20 to 45 degrees.               |  <img src="docs/images/tasks/stairsup.png" width="250"/>  |
|         [Down the Stairs](cfg/task/benchmark/StairsDown.yaml)         |             Simulate the tracked vehicle descending stairs. The width of each step of the stairs ranges from 0.2$m$ to 0.3$m$, and the slope of the stairs ranges from 20 to 45 degrees.              | <img src="docs/images/tasks/stairsdown.png" width="250"/> |
|            [Mixed Terrains](cfg/task/benchmark/Mixed.yaml)            |             Simulate the tracked vehicle navigating through various complex terrains, with the successful completion of the task defined when the vehicle reaches the specified endpoint.             |   <img src="docs/images/tasks/mixed.png" width="250"/>    |



## Install

### Setup
When using FTR, you need to configure some environment variables. Please execute the following code in the root directory of the project.
```shell
source shell/setup/ptask.sh
```
Or you can add the following code to *.bashrc*
```shell
export PTASK_HOME=<your FTR directory>
source ${PTASK_HOME}/shell/setup/ptask.sh
```

### Isaac Sim
The simulation platform used in this project is Isaac Sim, and version 2023.1.1 is recommended. For detailed installation instructions, please refer to the official website installation address.

[download and install Isaac Sim](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_workstation.html)

If there are multiple Isaac Sims on your computer, please modify the environment variable *ISAACSIM_HOME* configured in the *shell/setup/ptask.sh* file.

```shell
export ISAACSIM_HOME=`realpath <Specify the path of Isaac Sim>`
```

### Conda
If you want to run the project in Conda, please configure the environment variable *PTASK_CONDA_NAME* to indicate the conda environment name you want to use.
```shell
export PTASK_CONDA_NAME=<your conda environment name>
```

### Install dependencies
If your configuration is successful, you can use the *isaac-python* and *isaac-pip* commands in bash.
Use isaac-pip to install dependencies
```shell
isaac-pip install -r ptask_requirements.txt
```



## Train

To start training, you can execute the following code

~~~shell
isaac-python -m ptask_ftr task=benchmark/StepsUp train=FTR/SAC headless=True num_envs=2 experiment='StepUp_SAC' rl_device='cuda:0' sim_device=cpu
~~~

If you want to test the training results, you can use the following command

~~~shell
isaac-python -m ptask_ftr task=benchmark/StepsUp train=FTR/SAC test=True checkpoint=<checkpoint file>
~~~

## Supported RL algorithms are listed below:

- Proximal Policy Optimization (PPO) ([Paper](https://arxiv.org/pdf/1707.06347.pdf), [Config](cfg/train/FTR/PPO.yaml))
- Trust Region Policy Optimization (TRPO) ([Paper](https://arxiv.org/pdf/1502.05477.pdf), [Config](cfg/train/FTR/TRPO.yaml))
- Twin Delayed DDPG (TD3) ([Paper](https://arxiv.org/pdf/1802.09477.pdf), [Config](cfg/train/FTR/TD3.yaml))
- Soft Actor-Critic (SAC) ([Paper](https://arxiv.org/pdf/1812.05905.pdf), [Config](cfg/train/FTR/SAC.yaml))
- Deep Deterministic Policy Gradient (DDPG) ([Paper](https://arxiv.org/pdf/1509.02971.pdf), [Config](cfg/train/FTR/DDPG.yaml))


## Enviroment Performance
<table>
    <tr>
        <th colspan="2">Up the Steps</th>
        <th colspan="2">Down the Steps</th>
    <tr>
    <tr>
        <td><img src="docs/images/train_ftr/stepsup.gif" align="middle" width="700"/></td>
        <td><img src="docs/images/benchmark/stepsup.png" align="middle" width="500"/></td>
        <td><img src="docs/images/train_ftr/stepsdown.gif" align="middle" width="700"/></td>
        <td><img src="docs/images/benchmark/stepsdown.png" align="middle" width="500"/></td>
    <tr>
    <tr>
        <th colspan="2">Cross the Uplifts</th>
        <th colspan="2">Through Single-sided Steps</th>
    <tr>
    <tr>
        <td><img src="docs/images/train_ftr/uplifts.gif" align="middle" width="700"/></td>
        <td><img src="docs/images/benchmark/uplifts.png" align="middle" width="500"/></td>
        <td><img src="docs/images/train_ftr/unisteps.gif" align="middle" width="700"/></td>
        <td><img src="docs/images/benchmark/unisteps.png" align="middle" width="500"/></td>
    <tr>
    <tr>
        <th colspan="2">Drive on Plum Piles</th>
        <th colspan="2">Drive on Wave Terrains  </th>
    <tr>
    <tr>
        <td><img src="docs/images/train_ftr/plumpiles.gif" align="middle" width="700"/></td>
        <td><img src="docs/images/benchmark/plumpiles.png" align="middle" width="500"/></td>
        <td><img src="docs/images/train_ftr/waves.gif" align="middle" width="700"/></td>
        <td><img src="docs/images/benchmark/waves.png" align="middle" width="500"/></td>
    <tr>
    <tr>
        <th colspan="2">Cross the Rails</th>
        <th colspan="2">Up the Stairs </th>
    <tr>
    <tr>
        <td><img src="docs/images/train_ftr/rails.gif" align="middle" width="700"/></td>
        <td><img src="docs/images/benchmark/rails.png" align="middle" width="500"/></td>
        <td><img src="docs/images/train_ftr/stairsup.gif" align="middle" width="700"/></td>
        <td><img src="docs/images/benchmark/stairsup.png" align="middle" width="500"/></td>
    <tr>
    <tr>
        <th colspan="2">Down the Stairs </th>
        <th colspan="2">Mixed Terrains </th>
    <tr>
    <tr>
        <td><img src="docs/images/train_ftr/stairsdown.gif" align="middle" width="700"/></td>
        <td><img src="docs/images/benchmark/stairsdown.png" align="middle" width="500"/></td>
        <td><img src="docs/images/train_ftr/mixed.gif" align="middle" width="700"/></td>
        <td><img src="docs/images/benchmark/mixed.png" align="middle" width="500"/></td>
    <tr>

    
</table>

## Research papers published using FTR-Bench

<details open>
<summary>(Click to Collapse)</summary>

- Research paper
  - [Geometry-Based Flipper Motion Planning for Articulated Tracked Robots Traversing Rough Terrain in Real-time](https://onlinelibrary.wiley.com/doi/abs/10.1002/rob.22236): Journal of Field Robotics, 2023,  https://doi.org/10.1002/rob.22236.
  - [Deep Reinforcement Learning for Flipper Control of Tracked Robots](https://arxiv.org/abs/2306.10352): ArXiv 2023, https://doi.org/10.48550/arXiv.2306.10352.
</details>



