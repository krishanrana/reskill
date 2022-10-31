#!/bin/bash

conda env create -f environment.yml
conda activate reskill_new

pip install -e .
conda install pillow==6.1







