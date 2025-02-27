# pushing

## Project Description

This project focuses on developing and evaluating a behavior cloning policy for a robotic hand tasked with pushing a box to a specified target. Leveraging a 2D action space, the project simplifies complex manipulation tasks into manageable actions. Our approach involves training a Behavior Cloning (BC) model based on demonstrations from a hardcoded expert policy, aiming to achieve high success rates in pushing tasks within a simulated environment. This README provides instructions for setting up the project environment, running the simulation, and executing the BC policy.

## Demonstrations

### Expert Policy Demonstration

Click on the image below to watch the expert policy demonstration video:

[![Watch the expert policy demonstration](https://img.youtube.com/vi/7yzIU9E_nwc/0.jpg)](https://youtu.be/7yzIU9E_nwc)

### BC-Policy Demonstration

Click on the image below to watch the BC-policy demonstration video:

[![Watch the BC-policy demonstration](https://img.youtube.com/vi/6sxFLG1gamM/0.jpg)](https://youtu.be/6sxFLG1gamM)

### BC-Policy Demonstration

Click on the image below to watch the BC-policy with vision based box pose estimation demonstration video:

[![Watch the BC-policy demonstration](https://img.youtube.com/vi/A-hoKTmK0vU/0.jpg)](https://youtu.be/A-hoKTmK0vU)

## Installation

Follow these steps to set up your environment and run the project:

### Clone the Repository

First, clone the project repository from GitLab:

```bash
git clone https://gitlab.lrz.de/smlr_ws2324/g3/pushing.git --single-branch --depth 1 -b main
```

### Clone the Repository
Next, pull the Docker image specifically prepared for this project:
```bash
docker image pull rezaarrazi/smlr_py38_new
```

### Prepare the Environment
Run the Docker container using the provided script:
```bash
cd pushing
./Docker/run_container.sh
```

### Run the Simulator
Open a new terminal window and launch the simulator environment:
```bash
cd pushing/environment/simulator/v0.9/Linux/ManipulatorEnvironment_Data/
./ManipulatorEnvironment.x86_64 
```

### Execute the Behavior Cloning Policy
Inside the Docker container, initiate the BC policy by running:
```bash
cd smlr
python3 agent/main.py
```

## Quickstart: from 0 to a trained BC policy.

**Step 1**: Do the installation steps in  [Installation](#Installation).

**Step 2**: Download the expert demenostration data [here](https://drive.google.com/drive/folders/1IH8XYPKUCio1pkSE9MuVJmSOBX8E2WiQ?usp=sharing)

**(Optional) Step 3**: Generate trajectories from expert policy:

Set the config in ./agent/config.py:

```python
'simulation_mode': 'expert',
'num_episode': 100,
'success_episode_only': True,
'allow_variable_horizon': False, # to maintain the same timesteps length
'allow_dead_zone': True,
'data_dir': "./agent/data/1_side_300_random",
```

Run the main.py script:
```bash
python3 agent/main.py
```

**Step 4**: Set the config in agent/config.py:

```python 
'simulation_mode': 'train_bc',
'data_dir': "./agent/data/1_side_300_random",
```

**Step 5**: Execute the Behavior Cloning Policy Training

Inside the Docker container, initiate the training by running:
```bash
python3 agent/main.py
```

**Step 6**: Evaluate Trained Policy
Set the config in ./agent/config.py:

```python
'simulation_mode': 'evaluate_bc',
'log_bc_dir': "./agent/logs/<YOUR-BC-LOG-DIR>/bc_policy",
```

Inside the Docker container, initiate the evaluation by running:
```bash
python3 agent/main.py
```

**You're done with Quickstart!**

## Evaluate using vision based pose estimation.
Set the config in agent/config.py:

```python 
config_dict = {
    'simulation_mode': 'evaluate_bc',
    'use_images':      True
}

localization_config_dict = {
    "use_localization" : True,
    "data_cfg" : "/home/smlr/agent/pose_estimation/singleshotpose/cfg/box_series.data",
    "model_cfg" : "/home/smlr/agent/pose_estimation/singleshotpose/cfg/yolo-pose.cfg",
    "weightfile" : "/home/smlr/agent/pose_estimation/singleshotpose/data/models/randomized_best.ckpt"
}
```

Inside the Docker container, initiate the evaluation by running:
```bash
python3 agent/main.py
```