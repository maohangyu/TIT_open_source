"""
highly based on https://github.com/kzl/decision-transformer/blob/master/gym/experiment.py
"""

import torch
from torch.utils.data import Dataset

import numpy as np
import pickle
import random

import os
import blosc
import gym
import babyai
import re


class SequenceDataset(Dataset):
    def __init__(self, config):
        super(SequenceDataset, self).__init__()
        self.device = config.get('device', 'cuda')
        self.env_name = config['env_name']
        
        self.env = gym.make(self.env_name[0], disable_env_checker=True)
        
        self.image_dim = self.env.observation_space['image'].shape
        self.act_dim = self.env.action_space.n
        
        dataset_path = config['dataset_path']
        self.demos = load_demos(dataset_path, config['data_name'], config['step_num'])

        self.max_ep_len = config['max_ep_len']
        self.vocab = Vocabulary(config['text_max_size'])
        # save all path information into separate lists
        self.images, self.traj_lens, self.returns = [], [], []
        for demo in self.demos:
            self.images.append(demo['image'])
            self.traj_lens.append(len(demo['action']))
            demo['reward'][-1] = max(0, 1 - 0.9 * (len(demo['reward']) / self.max_ep_len))
            self.returns.append(sum(demo['reward']))
        self.mission, self.mission_mask = preprocess_texts([demo['mission'] for demo in self.demos], self.vocab)
        self.traj_lens, self.returns = np.array(self.traj_lens), np.array(self.returns)
        
        # used for input normalization
        self.images = np.concatenate(self.images, axis=0)

        self.K = config['K']
        self.pct_traj = config.get('pct_traj', 1.)

        # only train on top pct_traj trajectories (for %BC experiment)
        num_timesteps = sum(self.traj_lens)
        num_timesteps = max(int(self.pct_traj * num_timesteps), 1)
        sorted_inds = np.argsort(self.returns)  # lowest to highest
        num_trajectories = 1
        timesteps = self.traj_lens[sorted_inds[-1]]
        ind = len(self.demos) - 2
        while ind >= 0 and timesteps + self.traj_lens[sorted_inds[ind]] <= num_timesteps:
            timesteps += self.traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        self.sorted_inds = sorted_inds[-num_trajectories:]
        
        # used to reweight sampling so we sample according to timesteps instead of trajectories
        self.p_sample = self.traj_lens[self.sorted_inds] / sum(self.traj_lens[self.sorted_inds])
        
    def __getitem__(self, index):
        traj = self.demos[int(self.sorted_inds[index])]
        start_t = random.randint(0, len(traj['action']) - 1)
        
        s = traj['image'][start_t: start_t + self.K]
        a = traj['action'][start_t: start_t + self.K]
        a = np.eye(self.act_dim)[a]
        r = traj['reward'][start_t: start_t + self.K]
        m = self.mission[int(self.sorted_inds[index])]
        m_mask = self.mission_mask[int(self.sorted_inds[index])]
        if 'terminal' in traj:
            d = traj['terminal'][start_t: start_t + self.K]
        else:
            d = traj['done'][start_t: start_t + self.K]
        timesteps = np.arange(start_t, start_t + s.shape[0])
        timesteps[timesteps >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff
        rtg = self.discount_cumsum(traj['reward'][start_t:], gamma=1.)[:s.shape[0] + 1].reshape(-1, 1)
        if rtg.shape[0] <= s.shape[0]:
            rtg = np.concatenate([rtg, np.zeros((1, 1))], axis=0)

        # padding and state + reward + rtg normalization
        tlen = s.shape[0]
        s = np.concatenate([np.zeros((self.K - tlen, *self.image_dim)), s], axis=0)
        m = np.expand_dims(m, 0).repeat(self.K, axis=0)
        m_mask = np.expand_dims(m_mask, 0).repeat(self.K, axis=0)
        a = np.concatenate([np.zeros((self.K - tlen, self.act_dim)), a], axis=0) # how to pad action
        r = np.concatenate([np.zeros((self.K - tlen, 1)), np.array(r).reshape(-1, 1)], axis=0)
        d = np.concatenate([np.ones((self.K - tlen)) * 2, d], axis=0)
        rtg = np.concatenate([np.zeros((self.K - tlen, 1)), rtg], axis=0)
        timesteps = np.concatenate([np.zeros((self.K - tlen)), timesteps], axis=0)
        mask = np.concatenate([np.zeros((self.K - tlen)), np.ones((tlen))], axis=0)
        
        s = torch.from_numpy(s).to(dtype=torch.float32, device=self.device)
        m = torch.from_numpy(m).to(dtype=torch.long, device=self.device)
        m_mask = torch.from_numpy(m_mask).to(dtype=torch.float32, device=self.device)
        a = torch.from_numpy(a).to(dtype=torch.float32, device=self.device)
        r = torch.from_numpy(r).to(dtype=torch.float32, device=self.device)
        d = torch.from_numpy(d).to(dtype=torch.long, device=self.device)
        rtg = torch.from_numpy(rtg).to(dtype=torch.float32, device=self.device)
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.long, device=self.device)
        mask = torch.from_numpy(mask).to(device=self.device)
        return s, m, m_mask, a, r, d, rtg, timesteps, mask
        
    def discount_cumsum(self, x, gamma=1.):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(len(x)-1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
        return discount_cumsum
    


def load_pickle(path, raise_not_found=True):
    try:
        return pickle.load(open(path, "rb"))
    except FileNotFoundError:
        if raise_not_found:
            raise FileNotFoundError("No demos found at {}".format(path))
        else:
            return []


def transform_demos(demos):
    new_demos = []
    for demo in demos:
        new_demo = {}
        mission = demo[0]
        all_images = demo[1]
        directions = demo[2]
        actions = demo[3]
        all_images = blosc.unpack_array(all_images)
        n_observations = all_images.shape[0]
        assert len(directions) == len(actions) == n_observations, "error transforming demos"
        new_demo['image'] = all_images
        new_demo['direction'] = directions
        new_demo['mission'] = mission
        new_demo['action'] = [a.value for a in actions]
        new_demo['done'] = [i == n_observations - 1 for i in range(n_observations)]
        new_demo['reward'] = [1 if i == n_observations - 1 else 0 for i in range(n_observations)]
        new_demos.append(new_demo)
    return new_demos


def load_demos(path, files, step_num=None):
    all_demos = []
    for f in files:
        demos = load_pickle(path+'/'+f+'.pkl')
        demos = transform_demos(demos)
        demos_step_num = [len(d['action']) for d in demos]
        print('{} has {} time steps'.format(f, sum(demos_step_num)))
        if step_num is not None:
            n = 0
            for i, l in enumerate(demos_step_num):
                n += l
                if n >= step_num:
                    break
            all_demos.extend(demos[:i+1])
        else:
            all_demos.extend(demos)
    return all_demos


class Vocabulary:
    def __init__(self, max_size):
        self.max_size = max_size
        self.vocab = {}

    def load_vocab(self, vocab):
        self.vocab = vocab

    def __getitem__(self, token):
        if not token in self.vocab.keys():
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]


def preprocess_texts(texts, vocab):
    var_indexed_texts = []
    max_text_len = 0

    for text in texts:
        tokens = re.findall("([a-z]+)", text.lower())
        var_indexed_text = np.array([vocab[token] for token in tokens])
        var_indexed_texts.append(var_indexed_text)
        max_text_len = max(len(var_indexed_text), max_text_len)

    indexed_texts = np.zeros((len(texts), max_text_len))
    texts_mask = np.zeros((len(texts), max_text_len))

    for i, indexed_text in enumerate(var_indexed_texts):
        indexed_texts[i, :len(indexed_text)] = indexed_text
        texts_mask[i, :len(indexed_text)] = 1

    return indexed_texts, texts_mask
