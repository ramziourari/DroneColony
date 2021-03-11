import argparse
import yaml
import os
import pandas as pd

from envs.unity_env import Unity3DEnv

from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from models.simple_model import SimpletModel

from ray.rllib.models import ModelCatalog
from ray.rllib.env.policy_client import PolicyClient

SERVER_ADDRESS = "130.83.247.113"
SERVER_PORT = 6000

parser = argparse.ArgumentParser()
parser.add_argument(
    "--game",
    type=str,
    default=None,
    help="The game executable to run as RL env. If not provided, uses local "
    "Unity3D editor instance.")
parser.add_argument(
    "--horizon",
    type=int,
    default=200,
    help="The max. number of `step()`s for any episode (per agent) before "
    "it'll be reset again automatically.")
parser.add_argument(
    "--server",
    type=str,
    default=SERVER_ADDRESS + ":" + str(SERVER_PORT),
    help="The Policy server's address and port to connect to from this client."
)
parser.add_argument(
    "--no-train",
    action="store_true",
    help="Whether to disable training (on the server side).")
parser.add_argument(
    "--inference-mode",
    type=str,
    default="local",
    choices=["local", "remote"],
    help="Whether to compute actions `local`ly or `remote`ly. Note that "
    "`local` is much faster b/c observations/actions do not have to be "
    "sent via the network.")
parser.add_argument(
    "--update-interval-local-mode",
    type=float,
    default=10.0,
    help="For `inference-mode=local`, every how many seconds do we update "
    "learnt policy weights from the server?")
parser.add_argument(
    "--stop-reward",
    type=int,
    default=999999,
    help="Stop once the specified reward is reached.")

parser.add_argument(
    "--graphics",
    type=bool,
    default=False,
    help="if False the binary will be displayed (set to False for faster process).")

parser.add_argument(
    "-f",
    "--config-file",
    default=None,
    type=str,
    help="If specified, use config options from this file. Note that this "
         "overrides any trial-specific options set via flags above.")


conf_channel = EngineConfigurationChannel()
train_channel = EnvironmentParametersChannel()

if __name__ == "__main__":

    ModelCatalog.register_custom_model(
        "simple_model", SimpletModel)

    args = parser.parse_args()
    if args.config_file:
        with open(args.config_file) as f:
            experiments = yaml.safe_load(f)
    for exp in experiments.values():
        client_config = exp["client-config"]
        env_params = exp["env_params"]
        port = exp["port"]
        server = exp["server-adress"]
    # Start the client for sending environment information (e.g. observations,
    # actions) to a policy server (listening on port 9900).
    client = PolicyClient(
        "http://" + server + ":" + str(port),
        inference_mode="local",
        update_interval=10,)

    returns = []
    episodes = 0
    next_step = 0
    # --- environment parameters ---
    for k, v in env_params.items():
        train_channel.set_float_parameter(key=k, value=v)

    conf_channel.set_configuration_parameters(time_scale=client_config["time-scale"])

    # Start and reset the actual Unity3DEnv (either already running Unity3D
    # editor or a binary (game) to be started automatically).

    game = os.path.join(os.path.abspath(os.getcwd()), client_config["game"])
    env = Unity3DEnv(file_name=client_config["game"], episode_horizon=client_config["episode-horizon"],
                     no_graphics=client_config["no-graphic"], side_channels=[conf_channel, train_channel])
    obs = env.reset()
    eid = client.start_episode(training_enabled=not args.no_train)

    # Keep track of the total reward per episode.
    total_rewards_this_episode = 0.0
    # Loop infinitely through the env.
    count = 0
    dfs = []
    while True:
        # Get actions from the Policy server given our current obs.
        actions = client.get_action(eid, obs)
        client.update_policy_weights
        # Apply actions to our env.
        obs, rewards, dones, infos = env.step(actions)
        total_rewards_this_episode += sum(rewards.values())

        # --- uncomment to save trajectories ---
        # saving history for results analysis
        # df = pd.DataFrame(obs)
        # dfs.append(df)
        # hist = pd.concat(dfs)
        # hist.to_hdf("traj.h5", key="df")
        # hist.to_csv("history.csv")

        # Log rewards and single-agent dones.
        client.log_returns(eid, rewards, infos, multiagent_done_dict=dones)
        # Check whether all agents are done and end the episode, if necessary.
        if dones["__all__"]:
            print("Episode done: Reward={}".format(total_rewards_this_episode))
            returns.append(total_rewards_this_episode)
            if total_rewards_this_episode >= args.stop_reward:
                quit(0)
            # End the episode and reset Unity Env.
            total_rewards_this_episode = 0.0
            client.end_episode(eid, obs)
            obs = env.reset()
            # Start a new episode.
            eid = client.start_episode(training_enabled=not args.no_train)
            count += 1
