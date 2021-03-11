import argparse
import os
import yaml
import pandas as pd

from envs.unity_env import Unity3DEnv
from models.simple_model import SimpletModel

import ray
from ray.tune import register_env
from ray.rllib.env.policy_server_input import PolicyServerInput
from ray.rllib.examples.env.random_env import RandomMultiAgentEnv
from ray.rllib.agents.ppo import ppo
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.models import ModelCatalog

SERVER_ADDRESS = "130.83.247.113"
SERVER_PORT = 9900
CHECKPOINT_FILE = "last_checkpoint_{}.out"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    type=str,
    default=None,
    choices=["3DBall", "SoccerStrikersVsGoalie", "Game_MultiDroneDiscrete_Sparse", "Game_SingleDroneDiscrete_Sparse",
             "Game_SingleDroneDiscrete_Sparse_CL_DR", "Game_SingleDroneDiscrete_Dense_CL_DR"],
    help="The name of the Env to run in the Unity3D editor. Either `3DBall` "
    "or `SoccerStrikersVsGoalie` (feel free to add more to this script!)")
parser.add_argument(
    "--port",
    type=int,
    default=SERVER_PORT,
    help="The Policy server's port to listen on for ExternalEnv client "
    "conections.")
parser.add_argument(
    "--checkpoint-freq",
    type=int,
    default=10,
    help="The frequency with which to create checkpoint files of the learnt "
    "Policies.")
parser.add_argument(
    "--no-restore",
    action="store_true",
    help="Whether to load the Policy "
    "weights from a previous checkpoint")

parser.add_argument(
    "-n",
    "--experiment-name",
    default="drone_exp",
    type=str,
    help="optional naming of experiment")
parser.add_argument(
    "--cpus",
    type=int,
    default=4,
    help="number of cpus for server.")
parser.add_argument(
    "-f",
    "--config-file",
    default=None,
    type=str,
    help="If specified, use config options from this file. Note that this "
         "overrides any trial-specific options set via flags above.")


def name_exp(env):
    from datetime import datetime
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    exp_name = env + "_" + date_time
    return exp_name


if __name__ == "__main__":

    ModelCatalog.register_custom_model(
        "simple_model", SimpletModel)

    args = parser.parse_args()

    if args.config_file:
        with open(args.config_file) as f:
            experiments = yaml.safe_load(f)
    else:
        # The entire config will be sent to connecting clients so they can
        # build their own samplers (and also Policy objects iff
        # `inference_mode=local` on clients' command line).
        experiments = {
            args.experiment_name: {
                "port": args.port,
                "env": args.env,
                "num_cpus":args.cpus,
                "config": {
                    # Use the connector server to generate experiences.
                    "input": (
                        lambda ioctx: PolicyServerInput(ioctx, SERVER_ADDRESS, args.port)
                    ),
                    # Use a single worker process (w/ SyncSampler) to run the server.
                    "num_workers": 0,
                    # Disable OPE, since the rollouts are coming from online clients.
                    "input_evaluation": [],
                    # Other settings.
                    "sample_batch_size": 64,
                    "train_batch_size": 256,
                    "rollout_fragment_length": 20,
                    "framework": "tf",
                },
            }
        }
    for exp in experiments.values():
        env = exp["env"]
        num_cpus = exp["num_cpus"]
    ray.init(local_mode=True, num_cpus=num_cpus)
    # Create a fake-env for the server. This env will never be used (neither
    # for sampling, nor for evaluation) and its obs/action Spaces do not
    # matter either (multi-agent config below defines Spaces per Policy).
    register_env("fake_unity", lambda c: RandomMultiAgentEnv(c))
    policies, policy_mapping_fn = \
        Unity3DEnv.get_policy_configs_for_game(env)

    # make possible to use multi-agents and extract config
    for exp in experiments.values():
        exp["config"].update({
            # Multi-agent setup for the particular env.
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn},
            "input": (
                lambda ioctx: PolicyServerInput(ioctx, SERVER_ADDRESS, exp["port"])),
        })
        config = exp["config"]
        # if yaml file given, get trainer config, sonst default
        if args.config_file:
            try:
                trainer_config = exp["PPO-config"]
                ppo.DEFAULT_CONFIG.update(trainer_config)
            except:
                pass
    from ray.rllib.agents.ppo import ppo
    PPOTrainer = build_trainer(
        name=name_exp(env),
        default_config=ppo.DEFAULT_CONFIG,
        default_policy=PPOTFPolicy,
        get_policy_class=ppo.get_policy_class,
        execution_plan=ppo.execution_plan,
        validate_config=ppo.validate_config)
    # Create the Trainer used for Policy serving.
    trainer = PPOTrainer(env="fake_unity", config=config)
    # Attempt to restore from checkpoint if possible.
    checkpoint_path = CHECKPOINT_FILE.format(env)
    if not args.no_restore and os.path.exists(checkpoint_path):
        checkpoint_path = open(checkpoint_path).read()
        print("Restoring from checkpoint path", checkpoint_path)
        trainer.restore(checkpoint_path)
    # Serving and training loop.
    count = 0
    while True:
        # Calls to train() will block on the configured `input` in the Trainer
        # config above (PolicyServerInput).
        output = trainer.train()
        if count % args.checkpoint_freq == 0:
            print("Saving learning progress to checkpoint file.")
            checkpoint = trainer.save()

            # --- uncomment to save NN weights ---
            # pd.Series(trainer.get_weights()).to_json(path_or_buf="weights_{}.txt".format(count), orient='values')

            # Write the latest checkpoint location to CHECKPOINT_FILE,
            # so we can pick up from the latest one after a server re-start.
            with open(checkpoint_path, "w") as f:
                f.write(checkpoint)
        count += 1
