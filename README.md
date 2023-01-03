# Residual Skill Policies: Learning an Adaptable Skill-based Action Space for Reinforcement Learning for Robotics

[![QUT Centre for Robotics Open Source](https://github.com/qcr/qcr.github.io/raw/master/misc/badge.svg)](https://qcr.github.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


[[Paper]](https://arxiv.org/pdf/2211.02231.pdf)
[[Project Page]](https://krishanrana.github.io/reskill)

<p align="center">
  <img src="/images/reskill.png" width="800" />
</p>
<p align="center">

Official PyTorch implementation for the publication Residual Skill Policies: Learning an Adaptable Skill-based Action Space for Reinforcement Learning for Robotics (CoRL 2022)

## Requirements

- python 3.7+
- mujoco 2.1
- Ubuntu 18.04

## Installation Instructions

To install MuJoCo follow the instructions [here](https://github.com/openai/mujoco-py).

Clone the repository

```
git clone https://github.com/krishanrana/reskill.git
```
Ensure [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) is installed and configured for your system.
Create a conda environment and install all required packages.

```
cd reskill
conda env create -f environment.yml
conda activate reskill_new
pip install -e .
cd reskill

```

## Data Collection and Training
To collect a dataset using the scripted controllers run the following command:
```
python data/collect_demos.py --num_trajectories 40000 --subseq_len 10 --task block
```
There are two sets of tasks `block` and `hook`
The dataset collected for the `block` tasks can be used to train a downstream RL agent in the `FetchPyramidStack-v0`, `FetchCleanUp-v0` and `FetchSlipperyPush-v0` environments.
The dataset collected for the `hook` task is used to train the downstream RL agent in the `FetchComplexHook-v0` environment.
We collect the demonstration data for the hook and block based environments in the `FetchHook-v0` and `FetchPlaceMultiGoal-v0` environments respectively.

You can alternatively download a pre-collected dataset from [here](https://drive.google.com/drive/folders/1yTr_6fc-sHXK_CZkm8QIRTV9VgWxKpOE) and place the unzipped `dataset` folder in the root of the repository.

To train the skill modules on the collected dataset run the following command:
```
python train_skill_modules.py --config_file block/config.yaml --dataset_name fetch_block_40000
```
To visualise the performance of the trained skill module run the following command:
```
python utils/test_skill_modules.py --dataset_name fetch_block_40000 --task block --use_skill_prior True
```

To train the ReSkill agent using the trained skill modules, run the following command:

```
python train_reskill_agent.py --config_file table_cleanup/config.yaml --datatset_name fetch_block_40000
```
  
## Logging
  
All results are logged using [Weights and Biases](https://wandb.ai). An account and initial login is required to initialise logging as described on thier website.

## Code Structure
```
reskill
   |-- data 			# collected demonstration data
   |-- reskill			# contains all executable code 
   |   |-- configs 		# all config files for experiments
   |   |   |-- rl  		# config files for rl experiements
   |   |   `-- skill_mdl	# config files for both skill vae and skill prior
   |   |-- data			# dataset specifc code for collection and loading
   |   |-- models		# holds all model classes that implement forward and loss
   |   |-- results		# stores all trained pytorch models for both skill and rl modules
   |   |-- rl			# all code related to rl
   |   |   |-- agents		# implements core algorithms for rl agents
   |   |   |-- envs		# defines the set of environments for data collection and training
   |   |   `-- utils		# utilities for multiprocessing and training distributed rl agents
   |   |-- utils		# general utilities for data management and testing trained modules
   |   |   `-- controllers 	# set of scripted controllers used for data collection
   |   `-- wandb		# logging data from wandb
```



## Citation

```
  @article{rana2022reskill,
    title={Residual Skill Policies: Learning an Adaptable Skill-based Action Space for Reinforcement Learning for Robotics},
    author={Rana, Krishan and Xu, Ming and Tidd, Brendan and Milford, Michael and S{\"u}nderhauf, Niko},
    journal={Conference on Robot Learning (CoRL) 2022},
    year={2022}
  }
```
## Troubleshooting

```
ImportError: cannot import name 'PILLOW_VERSION' from 'PIL'
```
This is due to a Pillow version mismatch between the versions installed by mujoco-py and pytorch. A compatible package that fixes the issue is:
```
conda install pillow=6.1
```
---
  
```
Creating window glfw
ERROR: GLEW initalization error: Missing GL version

```
Error when rendering a MuJoCo window for visualisation. This can be fixed by exporting the required package:
```
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```


