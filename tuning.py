"""Population based training with PPO."""

import os
import random
import argparse

from envs.unity_env import Unity3DEnv

import ray
from ray.tune import run, sample_from, register_env
from ray.tune.schedulers import PopulationBasedTraining

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num-cpus",
    type=int,
    default=4,
    help="number of cpus to use per node")
parser.add_argument(
    "--redis-password",
    type=str,
    default=None,
    help="redis_password")

if __name__ == "__main__":

    args = parser.parse_args()

    # --- register env ---
    policies, policy_mapping_fn = \
        Unity3DEnv.get_policy_configs_for_game("Game_MultiDroneDenseContinuous_neu")

    # --- extract unity game ---
    cwd = os.path.abspath(os.getcwd())
    game = "/envs/Game_AtoB.x86_64"  # change this to your custom environment
    game = cwd + game  # can try os.join() instead

    register_env(
        "unity3d",
        lambda c: Unity3DEnv(
            file_name=game,
            no_graphics=True,
            episode_horizon=500))

    # Postprocess the perturbed config to ensure it's still valid
    def explore(config):
        # ensure we collect enough timesteps to do sgd
        if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
            config["train_batch_size"] = config["sgd_minibatch_size"] * 2
        # ensure we run at least one sgd iter
        if config["num_sgd_iter"] < 1:
            config["num_sgd_iter"] = 1
        return config

    pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        metric="episode_reward_mean",
        mode="max",
        perturbation_interval=120,
        resample_probability=0.25,
        # Specifies the mutations of these hyperparams
        hyperparam_mutations={
            "lambda": lambda: random.uniform(0.9, 1.0),
            "clip_param": lambda: random.uniform(0.01, 0.5),
            "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "num_sgd_iter": lambda: random.randint(1, 30),
            "sgd_minibatch_size": lambda: random.randint(128, 16384),
            "train_batch_size": lambda: random.randint(2000, 160000),
        },
        custom_explore_fn=explore)

    # --- for cluster training ---
    # redis_password = args.redis_password
    # num_cpus = int(args.num_cpus)
    # ray.init(address=os.environ["ip_head"], redis_password=redis_password)
    # print("Nodes in the Ray cluster:")
    # print(ray.nodes())

    ray.init()  # comment this when on cluster

    run(
        "PPO",
        name="pbt_AtoB",
        config={
            "env": "unity3d",
            "kl_coeff": 1.0,
            "num_workers": 4,
            "num_gpus": 0,
            "model": {

                "free_log_std": True
            },
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn},
            # These params are tuned from a fixed starting value.
            "lambda": 0.95,
            "clip_param": 0.2,
            "lr": 1e-4,
            # These params start off randomly drawn from a set.
            "num_sgd_iter": sample_from(
                lambda spec: random.choice([10, 20, 30])),
            "sgd_minibatch_size": sample_from(
                lambda spec: random.choice([128, 512, 2048])),
            "train_batch_size": sample_from(
                lambda spec: random.choice([10000, 20000, 40000]))
        },
        local_dir=os.path.join(os.path.abspath(os.getcwd()), "results/"),
        scheduler=pbt,
        num_samples=8,
        verbose=1,
        resume=False,
        checkpoint_freq=90)

    ray.shutdown()
