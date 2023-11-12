import datetime
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from generalization_benchmark.utils.helper_functions import gym_environment_creator

halfcheetah_d4rl, dataset, obs_prod, obs_shape, action_shape, action_prod, max_action = gym_environment_creator(
    'halfcheetah-medium-v2')


def jax_to_torch(jax_array):
    return torch.from_numpy(np.array(jax_array)).float().to("cuda")


def plotter(result_df, title, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(result_df['timestep'], result_df['normalized_reward'])
    plt.xlabel('Timestep')
    plt.ylabel('Normalized reward')
    plt.title(title)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def evaluate_policy(agent_policy, eval_episodes=10,
                    render=False, env=halfcheetah_d4rl,
                    remove_first_obs=False, convert_jax_to_torch=False,
                    seed: int = 42) -> pd.DataFrame:
    """
    Evaluate the policy given the agent policy, environment, and other parameters
    :param agent_policy: policy of the agent
    :param eval_episodes: number of episodes to evaluate
    :param render: whether to render the environment
    :param env:  to evaluate the policy
    :param remove_first_obs: remove the first obs if it is a CARL environment (redundant obs)
    :param convert_jax_to_torch: boolean to convert jax array to torch array
    :param seed: seed for reproducibility
    :return: dataframe of the evaluation results
    """
    # Set seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Reset the observations
    obs = env.reset(seed=seed)

    # Check if it needs to be converted to torch
    if convert_jax_to_torch:
        obs = env.reset()[0]['obs']
        obs = jax_to_torch(obs)
    # Check if first observation needs to be removed
    obs = obs[1:] if remove_first_obs else obs

    # Initiate variables
    eval_ep_info_buffer = []
    num_episodes = 0
    episode_reward = 0
    episode_length = 0

    # Iterate through the episodes
    with tqdm(total=eval_episodes, dynamic_ncols=True) as pbar:
        # Iterate through the episodes until episode length 1000 is reached
        while num_episodes < eval_episodes:
            # For each episode, reset the reward buffer
            reward_buffer = []
            # Select action
            action = agent_policy.select_action(obs.reshape(1, -1), deterministic=True)
            # Take a step
            try:
                next_obs, reward, done, _ = env.step(action.flatten())
            except ValueError as e:
                next_obs, reward, done, truncated, _ = env.step(action.flatten())

            # Append reward to reward buffer
            reward_buffer.append(reward)
            # Update episode reward and episode length
            episode_reward += reward
            episode_length += 1
            # Check if the new obs needs to be converted to torch
            if convert_jax_to_torch:
                next_obs = next_obs['obs']
                next_obs = jax_to_torch(next_obs)
            # Check if first observation needs to be removed
            next_obs = next_obs[1:] if remove_first_obs else next_obs
            obs = next_obs

            # End the episode if done or episode length is 1000
            if done or episode_length == 1000:
                eval_ep_info_buffer.append(
                    {"timestep": num_episodes,
                     "episode_reward": episode_reward,
                     "normalized_reward": halfcheetah_d4rl.get_normalized_score(episode_reward) * 100,
                     "episode_length": episode_length,
                     "mean_reward": round((episode_reward / episode_length), 2),
                     "std_reward": np.round(np.std(reward_buffer), 5),
                     })
                # Increment the number of episodes
                num_episodes += 1
                # Update the progress bar
                pbar.update(1)
                # Reset the episode (rewrite the code)
                obs = env.reset()
                # Set the seed again
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                np.random.seed(seed)
                random.seed(seed)
                # Check if it needs to be converted to torch
                if convert_jax_to_torch:
                    obs = env.reset()[0]['obs']
                    obs = jax_to_torch(obs)
                # Check if first observation needs to be removed
                obs = obs[1:] if remove_first_obs else obs
                # Reset the episode reward and episode length
                episode_reward, episode_length = 0, 0

            if render:
                env.render()
    # Convert the result buffer to a dataframe
    result_df = pd.DataFrame(eval_ep_info_buffer)
    return result_df


# %%
def benchmark(datasets, policies, seeds, save_dir, eval_episodes: int = 10):
    timestamp = datetime.datetime.now().strftime("%d_%m%_y_%H%M%S")

    for dataset_key, dataset_value in datasets.items():
        for policy_key, policy_value in policies.items():
            for seed in seeds:
                print(f"Running policy {policy_key.upper()} on dataset {dataset_key} with seed {seed}")
                # Check if dataset is a CARL environment
                if type(dataset_value.reset()) != np.ndarray:
                    convert_jax_to_torch = True
                else:
                    convert_jax_to_torch = False
                if dataset_key == "crippled_cheetah":
                    remove_first_obs = True
                else:
                    remove_first_obs = False
                # Evaluate policy on dataset with seed
                result_df = evaluate_policy(agent_policy=policy_value, eval_episodes=eval_episodes, env=dataset_value,
                                            convert_jax_to_torch=convert_jax_to_torch,
                                            remove_first_obs=remove_first_obs, seed=seed)

                # Create directory for saving results where the root directory is the timestamp
                timestamp_dir = os.path.join(save_dir, timestamp)
                policy_dir = os.path.join(timestamp_dir, policy_key)
                seed_dir = os.path.join(policy_dir, f"seed_{seed}")
                dataset_dir = os.path.join(seed_dir, dataset_key)
                os.makedirs(dataset_dir, exist_ok=True)
                print(f"Saving results to {dataset_dir}")
                # Save results as CSV file
                result_file = os.path.join(dataset_dir, "results.csv")
                result_df.to_csv(result_file, index=False)
    print("Benchmarking completed successfully!")
