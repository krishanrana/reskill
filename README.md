# Residual Skill Policies: Learning an Adaptable Skill-based Action Space for Reinforcement Learning for Robotics

[![QUT Centre for Robotics Open Source](https://github.com/qcr/qcr.github.io/raw/master/misc/badge.svg)](https://qcr.github.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


[[Paper]](https://openreview.net/pdf?id=BVZdCKCy3W)
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

```

## Training Commands
To collect a dataset using the scripted controllers run the following commands:
```
cd reskill/data
python collect_demos.py --num_trajectories 40000 --subseq_len 10 --task block
```
There are two sets of tasks `block` and `hook`
The dataset collected for the `block` tasks can be used to train a downstream RL agent in the `FetchPyramidStack-v0`, `FetchCleanUp-v0` and `FetchSlipperyPush-v0` environments.
The dataset collected for the `hook` task is used to train the downstream RL agent in the `FetchComplexHook-v0` environment.

To train the skill modules on the collected dataset run the following command:
```
python train_skill_modules.py --config_file config.yaml --dataset_name fetch_block_40000
```
To train the ReSkill agent using the trained skill modules, run the following command:

```
python train_reskill_agent.py --config_file config.yml --datatset_name fetch_block_40000
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
