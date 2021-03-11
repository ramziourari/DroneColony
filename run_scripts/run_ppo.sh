#!/bin/bash
#SBATCH -J 24_cpu_ppo-2-po-gs-2D
#SBATCH -n 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1GB
#SBATCH -t 00:10:00    
#SBATCH -e /work/scratch/bj19ihup/test.err.%j
#SBATCH -o /work/scratch/bj19ihup/test.out.%j
#SBATCH --mail-user=ramzi.ourari@tu-darmstadt.de
#SBATCH --mail-type=ALL
# -------------------------------
source anaconda3/etc/profile.d/conda.sh
conda activate swarm
cd /home/bj19ihup/dronecolony

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

node1=${nodes_array[0]}

ip_prefix=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --ip-address) # Making address
suffix=':6379'
ip_head=$ip_prefix$suffix
redis_password=$(uuidgen)

export ip_head # Exporting for latter access by trainer.py

srun --nodes=1 --ntasks=1 -w $node1 ray start --block --head --redis-port=6379 --redis-password=$redis_password & # Starting the head
sleep 5
# Make sure the head successfully starts before any worker does, otherwise
# the worker will not be able to connect to redis. In case of longer delay,
# adjust the sleeptime above to ensure proper order.

for ((  i=1; i<=$worker_num; i++ ))
do
  node2=${nodes_array[$i]}
  srun --nodes=1 --ntasks=1 -w $node2 ray start --block --address=$ip_head --redis-password=$redis_password & # Starting the workers
  # Flag --block will keep ray process alive on each compute node.
  sleep 5
done

python -u tuning.py --redis-password $redis_password --num-cpus 4
