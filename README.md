# Residual Skill Policies: Learning an Adaptable Skill-based Action Space for Reinforcement Learning for Robotics

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

## Citation

```
  @article{rana2022reskill,
    title={Residual Skill Policies: Learning an Adaptable Skill-based Action Space for Reinforcement Learning for Robotics},
    author={Rana, Krishan and Xu, Ming and Tidd, Brendan and Milford, Michael and S{\"u}nderhauf, Niko},
    journal={Conference on Robot Learning (CoRL) 2022},
    year={2022}
  }
```
