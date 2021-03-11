#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate swarm
cd ~/dronecolony/
python ppo_client.py --config-file experiments/PPO_AtoB.yaml
