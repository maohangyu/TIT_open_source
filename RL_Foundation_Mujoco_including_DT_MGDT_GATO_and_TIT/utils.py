"""
highly based on https://github.com/kzl/decision-transformer/blob/master/gym/experiment.py
"""

import torch
from torch.utils.data import Dataset

import gym
import d4rl
import numpy as np
import pickle
import random


class SequenceDataset(Dataset):
    def __init__(self, config):
        super(SequenceDataset, self).__init__()
        self.device = config.get('device', 'cuda')
        self.env_name = config['env_name']
        
        self.env = gym.make(self.env_name)
        self.max_ep_len = config['max_ep_len']
        self.scale = config['scale']
        self.reward_scale = config['reward_scale']
        
        self.state_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        
        dataset_path = 'data/{}.pkl'.format(config['data_name'])
        with open(dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)

        # save all path information into separate lists
        self.is_delayed_reward = config['delayed_reward']
        self.states, self.traj_lens, self.returns = [], [], []
        for path in self.trajectories:
            if self.is_delayed_reward:  # delayed: all rewards moved to end of trajectory
                path['rewards'][-1] = path['rewards'].sum()
                path['rewards'][:-1] = 0.
            self.states.append(path['observations'])
            self.traj_lens.append(len(path['observations']))
            self.returns.append(path['rewards'].sum())
        self.traj_lens, self.returns = np.array(self.traj_lens), np.array(self.returns)
        
        # used for input normalization
        self.states = np.concatenate(self.states, axis=0)
        self.state_mean, self.state_std = np.mean(self.states, axis=0), np.std(self.states, axis=0) + 1e-6

        self.K = config['K']
        self.pct_traj = config.get('pct_traj', 1.)

        # only train on top pct_traj trajectories (for %BC experiment)
        num_timesteps = sum(self.traj_lens)
        num_timesteps = max(int(self.pct_traj * num_timesteps), 1)
        sorted_inds = np.argsort(self.returns)  # lowest to highest
        num_trajectories = 1
        timesteps = self.traj_lens[sorted_inds[-1]]
        ind = len(self.trajectories) - 2
        while ind >= 0 and timesteps + self.traj_lens[sorted_inds[ind]] <= num_timesteps:
            timesteps += self.traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        self.sorted_inds = sorted_inds[-num_trajectories:]
        
        # used to reweight sampling so we sample according to timesteps instead of trajectories
        self.p_sample = self.traj_lens[self.sorted_inds] / sum(self.traj_lens[self.sorted_inds])
        
    def __getitem__(self, index):
        traj = self.trajectories[int(self.sorted_inds[index])]
        start_t = random.randint(0, traj['rewards'].shape[0] - 1)
        
        s = traj['observations'][start_t: start_t + self.K]
        a = traj['actions'][start_t: start_t + self.K]
        r = traj['rewards'][start_t: start_t + self.K].reshape(-1, 1)
        if 'terminals' in traj:
            d = traj['terminals'][start_t: start_t + self.K]
        else:
            d = traj['dones'][start_t: start_t + self.K]
        timesteps = np.arange(start_t, start_t + s.shape[0])
        timesteps[timesteps >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff
        rtg = self.discount_cumsum(traj['rewards'][start_t:], gamma=1.)[:s.shape[0] + 1].reshape(-1, 1)
        if rtg.shape[0] <= s.shape[0]:
            rtg = np.concatenate([rtg, np.zeros((1, 1))], axis=0)

        # padding and state + reward + rtg normalization
        tlen = s.shape[0]
        s = np.concatenate([np.zeros((self.K - tlen, self.state_dim)), s], axis=0)
        s = (s - self.state_mean) / self.state_std
        a = np.concatenate([np.ones((self.K - tlen, self.act_dim)) * -10., a], axis=0)
        r = np.concatenate([np.zeros((self.K - tlen, 1)), r], axis=0)
        r = -1 + 2 * (r - self.reward_scale[0]) / (self.reward_scale[1] - self.reward_scale[0])
        d = np.concatenate([np.ones((self.K - tlen)) * 2, d], axis=0)
        rtg = np.concatenate([np.zeros((self.K - tlen, 1)), rtg], axis=0) / self.scale
        timesteps = np.concatenate([np.zeros((self.K - tlen)), timesteps], axis=0)
        mask = np.concatenate([np.zeros((self.K - tlen)), np.ones((tlen))], axis=0)
        
        s = torch.from_numpy(s).to(dtype=torch.float32, device=self.device)
        a = torch.from_numpy(a).to(dtype=torch.float32, device=self.device)
        r = torch.from_numpy(r).to(dtype=torch.float32, device=self.device)
        d = torch.from_numpy(d).to(dtype=torch.long, device=self.device)
        rtg = torch.from_numpy(rtg).to(dtype=torch.float32, device=self.device)
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.long, device=self.device)
        mask = torch.from_numpy(mask).to(device=self.device)
        return s, a, r, d, rtg, timesteps, mask
        
    def discount_cumsum(self, x, gamma=1.):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0]-1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
        return discount_cumsum
    
