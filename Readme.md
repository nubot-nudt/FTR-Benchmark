```
 _______ _________ _______         ______   _______  _______          
(  ____ \\__   __/(  ____ )       (  ___ \ (  ____ \(  ____ \|\     /|
| (    \/   ) (   | (    )|       | (   ) )| (    \/| (    \/| )   ( |
| (__       | |   | (____)| _____ | (__/ / | (__    | |      | (___) |
|  __)      | |   |     __)(_____)|  __ (  |  __)   | |      |  ___  |
| (         | |   | (\ (          | (  \ \ | (      | |      | (   ) |
| )         | |   | ) \ \__       | )___) )| (____/\| (____/\| )   ( |
|/          )_(   |/   \__/       |/ \___/ (_______/(_______/|/     \|
```

# Introduce

This project uses the NuBot rescue robot as the platform and obstacles crossing as the task. It builds a reinforcement learning training system in Isaac Sim and implements commonly used reinforcement learning algorithms.

![](docs/images/out.gif)

# Install

## Isaac Sim
The simulation platform used in this project is Isaac Sim. The installation address of the official website is as follows

[download and install Isaac Sim](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_workstation.html)

## Python
After Isaac Sim is installed successfully, there is a file named python.sh in its installation directory. This is a python that comes with Isaac Sim. First, you need to configure the built-in Python environment of Isaac Sim. Here, name this python isisaac-python. If you are using bash, then add the following code to the last line of **.bashrc**

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
