#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate swarm
cd ~/dronecolony/
python ppo_server.py --config-file experiments/PPO_AtoB.yaml