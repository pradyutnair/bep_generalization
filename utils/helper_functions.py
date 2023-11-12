# %%
import argparse
import random
import warnings
from typing import Literal

import gym
import numpy as np
import torch

warnings.filterwarnings('ignore')

# seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def jax_to_torch(jax_array):
    return torch.from_numpy(np.array(jax_array)).float().to("cuda")


# %%
def gym_environment_creator(env_name: str = 'halfcheetah-medium-v2') -> (gym.Env, int, int, int):
    """
    Create gym environment
    :param env_name: string of the environment name
    :return:
    """
    # Create environment
    env = gym.make(env_name)
    dataset = env.get_dataset()
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    max_action = env.action_space.high[0]

    # Create product
    obs_prod = np.prod(obs_shape)
    action_prod = np.prod(action_shape)

    return env, dataset, obs_prod, obs_shape, action_shape, action_prod, max_action


# %%
halfcheetah_d4rl, dataset, obs_prod, obs_shape, action_shape, action_prod, max_action = gym_environment_creator(
    'halfcheetah-medium-v2')

print(f"Environment: halfcheetah-medium-v2")
print(f"Observation shape: {obs_shape}")
print(f"Action shape: {action_shape}")
print(f"Action prod: {action_prod}")
print(f"Max action: {max_action}")


# %%
def args_generator(algo_name: Literal['mopo', 'combo'] = 'mopo', task_name: str = 'halfcheetah-medium-v2'):
    template_args = {
        "mopo": {
            "algo_name": 'mopo',
            "hidden_dims": [256, 256],
        },
        "combo": {
            "algo_name": "combo",
            "hidden_dims": [256, 256, 256],
            "cql_weight": 0.5,
            "temperature": 1.0,
            "max_q_backup": False,
            "deterministic_backup": True,
            "with_lagrange": False,
            "lagrange_threshold": 10.0,
            "cql_alpha_lr": 3e-4,
            "num_repeat_actions": 10,
            "uniform_rollout": False,
            "rho_s": "mix",
        }
    }

    template_args[algo_name]["task"] = task_name
    template_args[algo_name]["seed"] = 1
    template_args[algo_name]["actor_lr"] = 1e-4
    template_args[algo_name]["critic_lr"] = 3e-4
    template_args[algo_name]["gamma"] = 0.99
    template_args[algo_name]["tau"] = 0.005
    template_args[algo_name]["alpha"] = 0.2
    template_args[algo_name]["alpha_lr"] = 1e-4
    template_args[algo_name]["auto_alpha"] = True
    template_args[algo_name]["target_entropy"] = None
    template_args[algo_name]["dynamics_lr"] = 1e-3
    template_args[algo_name]["dynamics_hidden_dims"] = [200, 200, 200, 200]
    template_args[algo_name]["dynamics_weight_decay"] = [2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4]
    template_args[algo_name]["n_ensemble"] = 7
    template_args[algo_name]["n_elites"] = 5
    template_args[algo_name]["rollout_freq"] = 1000
    template_args[algo_name]["rollout_batch_size"] = 50000
    template_args[algo_name]["rollout_length"] = 5
    template_args[algo_name]["model_retain_epochs"] = 5
    template_args[algo_name]["load_dynamics_path"] = None
    template_args[algo_name]["epoch"] = 1000
    template_args[algo_name]["step_per_epoch"] = 1000
    template_args[algo_name]["eval_episodes"] = 10
    template_args[algo_name]["batch_size"] = 256
    template_args[algo_name]['penalty_coef'] = 0.5
    template_args[algo_name]["real_ratio"] = 0.05
    template_args[algo_name]["device"] = "cuda"

    # Halfcheetah specific settings
    template_args[algo_name]["obs_shape"] = obs_shape
    template_args[algo_name]["action_dim"] = action_prod
    template_args[algo_name]["max_action"] = max_action

    argument = template_args[algo_name]
    return argparse.Namespace(**argument)

# %%
