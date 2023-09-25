```
  ______ _______ _____        ____                  _     
 |  ____|__   __|  __ \      |  _ \                | |    
 | |__     | |  | |__) |_____| |_) | ___ _ __   ___| |__  
 |  __|    | |  |  _  /______|  _ < / _ \ '_ \ / __| '_ \ 
 | |       | |  | | \ \      | |_) |  __/ | | | (__| | | |
 |_|       |_|  |_|  \_\     |____/ \___|_| |_|\___|_| |_|                                                                       
```

# Introduce

This project uses the NuBot rescue robot as the platform and obstacles crossing as the task. It builds a reinforcement learning training system in Isaac Sim and implements commonly used reinforcement learning algorithms.

![](docs/images/out.gif)

# Install

## Isaac Sim
The simulation platform used in this project is Isaac Sim. The installation address of the official website is as follows

[download and install Isaac Sim](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_workstation.html)

## Python
After Isaac Sim is installed successfully, there is a file named python.sh in its installation directory. This is a python that comes with Isaac Sim. First, you need to configure the built-in Python environment of Isaac Sim. Here, name this python isaac-python. If you are using bash, then add the following code to the last line of **.bashrc**

```shell
alias 'isaac-python=~/.local/share/ov/pkg/isaac_sim-*/python.sh'
```
## Install dependencies

```shell
isaac-python -m pip install -r requirements_isaac.txt
```
Related dependencies github
* [rl-games](https://github.com/Denys88/rl_games)
* [omniisaacgymenvs](https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs)

# Train

Execute the following code directly in the project root directory to start training

```shell
./shell/benchmark/train_args -a DDPG -s batten
```
Among them, the -a parameter indicates the training algorithm, including PPO, SAC, TRPO, DDPG and TD3, and the -s parameter indicates the map file to be used, including batten, flat, plum, sdown and sup.

# Supported RL algorithms are listed below:

- [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf)
- [Trust Region Policy Optimization (TRPO)](https://arxiv.org/pdf/1502.05477.pdf)
- [Twin Delayed DDPG (TD3)](https://arxiv.org/pdf/1802.09477.pdf)
- [Soft Actor-Critic (SAC)](https://arxiv.org/pdf/1812.05905.pdf)
- [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/pdf/1509.02971.pdf)
- [Dueling Double DQN (D3QN)](https://arxiv.org/pdf/1812.05905.pdf)

# Research papers published using FTR-Bench

<details open>
<summary>(Click to Collapse)</summary>

- Research paper
  - [Geometry-Based Flipper Motion Planning for Articulated Tracked Robots Traversing Rough Terrain in Real-time](https://onlinelibrary.wiley.com/doi/abs/10.1002/rob.22236): Journal of Field Robotics, 2023,  https://doi.org/10.1002/rob.22236.
  - [Deep Reinforcement Learning for Flipper Control of Tracked Robots](https://arxiv.org/abs/2306.10352): ArXiv 2023, https://doi.org/10.48550/arXiv.2306.10352.
<details close>



