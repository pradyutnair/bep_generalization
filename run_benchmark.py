import os

from generalization_benchmark.env.carl_cheetah import carl_cheetah_environment
from generalization_benchmark.env.cripple_cheetah import CrippleHalfCheetahEnv
from generalization_benchmark.model.combo_agent import combo
from generalization_benchmark.model.mopo_agent import mopo
from generalization_benchmark.utils.evaluation import benchmark
from generalization_benchmark.utils.helper_functions import args_generator, gym_environment_creator

# %%
# Generate arguments
print(f"Generating arguments...")
mopo_args = args_generator()
combo_args = args_generator(algo_name='combo')
# %%
# Create environments
print(f"Creating environments...")
halfcheetah_d4rl, dataset, obs_prod, obs_shape, action_shape, action_prod, max_action = gym_environment_creator(
    'halfcheetah-medium-v2')
# %%
# Create policies
print(f"Creating policies...")
# Load MOPO
print(f"Loading MOPO...")
mopo_policy_path = os.path.join(os.getcwd(), 'pretrained_models/halfcheetah-medium-v2/mopo/seed_3_timestamp_23-0103'
                                             '-130104/checkpoint/policy.pth')
mopo_dynamics_path = os.path.join(os.getcwd(), 'pretrained_models/halfcheetah-medium-v2/mopo/seed_2_timestamp_23-0103'
                                               '-130105/model/')

mopo_policy, policy_trainer = mopo(saved_policy=mopo_policy_path, saved_dynamics=mopo_dynamics_path)
print(f"MOPO loaded!")

# %%
# Load COMBO
print(f"Loading COMBO...")
# Load pre-trained COMBO policy
combo_policy_path = os.path.join(os.getcwd(), '../pretrained_models/halfcheetah-medium-v2/combo/seed_3_timestamp_23'
                                              '-0506-040857/checkpoint/policy.pth')
combo_dynamics_path = os.path.join(os.getcwd(), '../pretrained_models/halfcheetah-medium-v2/combo/seed_3_timestamp_23'
                                                '-0506-040857/model/')

combo_policy, combo_policy_trainer = combo(saved_policy=combo_policy_path, saved_dynamics=combo_dynamics_path)
print(f"COMBO loaded!")
# %%
# Load environments
print(f"Loading environments...")
modified_cheetah = carl_cheetah_environment()
crippled_cheetah = CrippleHalfCheetahEnv()
# %%
# Evaluate policies
print(f"Evaluating policies...")
datasets = {
    "halfcheetah-medium-v2": halfcheetah_d4rl,
    "modified_cheetah": modified_cheetah,
    "crippled_cheetah": crippled_cheetah
}
policies = {
    "mopo": mopo_policy,
    "combo": combo_policy
}

seeds = [1, 2, 3]
save_dir = "./eval_results"

if __name__ == '__main__':
    benchmark(datasets, policies, seeds, save_dir, eval_episodes=1)



