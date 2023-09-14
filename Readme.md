```
  _   _         ____          _   
 | \ | |       |  _ \        | |  
 |  \| | _   _ | |_) |  ___  | |_ 
 | . ` || | | ||  _ <  / _ \ | __|
 | |\  || |_| || |_) || (_) || |_ 
 |_| \_| \__,_||____/  \___/  \__|
                                  
  _____                    _                   
 |  __ \                  | |                  
 | |__) |_   _  _ __ ___  | |__    __ _   __ _ 
 |  ___/| | | || '_ ` _ \ | '_ \  / _` | / _` |
 | |    | |_| || | | | | || |_) || (_| || (_| |
 |_|     \__,_||_| |_| |_||_.__/  \__,_| \__,_|
                                                                    
  ____                      _                              _    
 |  _ \                    | |                            | |   
 | |_) |  ___  _ __    ___ | |__   _ __ ___    __ _  _ __ | | __
 |  _ <  / _ \| '_ \  / __|| '_ \ | '_ ` _ \  / _` || '__|| |/ /
 | |_) ||  __/| | | || (__ | | | || | | | | || (_| || |   |   < 
 |____/  \___||_| |_| \___||_| |_||_| |_| |_| \__,_||_|   |_|\_\ 
                                           
```

# Introduce

This project uses the NuBot rescue robot as the platform and obstacles crossing as the task. It builds a reinforcement learning training system in Isaac Sim and implements commonly used reinforcement learning algorithms.

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
