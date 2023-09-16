```
                                                                                                               
88888888888  888888888888  88888888ba           88888888ba                                        88           
88                88       88      "8b          88      "8b                                       88           
88                88       88      ,8P          88      ,8P                                       88           
88aaaaa           88       88aaaaaa8P'          88aaaaaa8P'   ,adPPYba,  8b,dPPYba,    ,adPPYba,  88,dPPYba,   
88"""""           88       88""""88'  aaaaaaaa  88""""""8b,  a8P_____88  88P'   `"8a  a8"     ""  88P'    "8a  
88                88       88    `8b  """"""""  88      `8b  8PP"""""""  88       88  8b          88       88  
88                88       88     `8b           88      a8P  "8b,   ,aa  88       88  "8a,   ,aa  88       88  
88                88       88      `8b          88888888P"    `"Ybbd8"'  88       88   `"Ybbd8"'  88       88  
                                                                                                                     
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
