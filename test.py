import gym
print(gym.__version__ )
env = gym.make('HalfCheetah-v2')
dataset = env.get_dataset()