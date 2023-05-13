"""
highly based on https://github.com/kzl/decision-transformer/blob/master/gym/data/download_d4rl_datasets.py
"""

import gym
import numpy as np
import os

import collections
import pickle

import d4rl  # Import required to register environments, you may need to also import the submodule

os.makedirs('./data', exist_ok=True)

dataset_name = [
    'hopper-medium-v2',
    'hopper-medium-replay-v2',
    'hopper-medium-expert-v2',
    'halfcheetah-medium-v2',
    'halfcheetah-medium-replay-v2',
    'halfcheetah-medium-expert-v2',
    'walker2d-medium-v2',
    'walker2d-medium-replay-v2',
    'walker2d-medium-expert-v2',
    'pen-cloned-v0',
    'door-cloned-v0',
    'relocate-cloned-v0',
    'hammer-cloned-v0',
    'antmaze-umaze-diverse-v0',
    'antmaze-medium-diverse-v0',
    'antmaze-large-diverse-v0',
]

for env_name in dataset_name:
    env = gym.make(env_name)
    dataset = env.get_dataset()

    N = dataset['rewards'].shape[0]
    max_reward = max(dataset['rewards'])
    min_reward = min(dataset['rewards'])
    data_ = collections.defaultdict(list)

    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    max_ep_len = 0

    episode_step = 0
    paths = []
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == 1000 - 1)
        for k in ['observations', 'actions', 'rewards', 'terminals']:
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            max_ep_len = max(episode_step, max_ep_len)
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            paths.append(episode_data)
            data_ = collections.defaultdict(list)
        episode_step += 1

    returns = np.array([np.sum(p['rewards']) for p in paths])
    num_samples = np.sum([p['rewards'].shape[0] for p in paths])
    print('env: {}'.format(env_name))
    print(f'Number of samples collected: {num_samples}')
    print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}, max_ep_len = {int(max_ep_len)}')
    print(f'Trajectory rewards: min = {min_reward}, max = {max_reward}')

    with open(f'./data/{env_name}.pkl', 'wb') as f:
        pickle.dump(paths, f)

