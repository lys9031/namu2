#!/bin/bash
cd ~
source ~/anaconda3/etc/profile.d/conda.sh
conda info
conda env list
#conda activate FT
conda deactivate
cd /home/kmj21/code/ForestAndTree_v3.0/monster//Monstermash-python-main/
python3 monstermash.py

