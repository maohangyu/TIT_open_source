"""
highly based on
https://github.com/kzl/decision-transformer/blob/master/gym/experiment.py#L166
https://github.com/kzl/decision-transformer/blob/master/gym/decision_transformer/evaluation/evaluate_episodes.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from utils import preprocess_texts


class Evaluation(object):
    def __init__(self, config, vocab):
        self.config = config
        self.device = config.get('device', 'cuda')
        self.env_name = config['env_name']
        self.num_eval_episodes = config['num_eval_episodes']
        self.model_type = config['model_type']
        
        self.env = gym.make(self.env_name[0], disable_env_checker=True)
        self.image_dim = self.env.observation_space['image'].shape
        self.act_dim = self.env.action_space.n
        
        self.vocab = vocab
        self.max_ep_len = config['max_ep_len']
        
    def evaluate_episode(self, model, env):
        self.env = gym.make(env, disable_env_checker=True)
        model.eval()
        model.to(device=self.device)

        state = self.env.reset()
        image = state['image']
        mission = state['mission']
        mission, _ = preprocess_texts([mission], self.vocab)

        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"
        images = torch.from_numpy(image).reshape(1, *self.image_dim).to(device=self.device, dtype=torch.float32)
        missions = torch.from_numpy(mission).to(device=self.device, dtype=torch.long)
        actions = torch.zeros((0, self.act_dim), device=self.device, dtype=torch.float32)
        rewards = torch.zeros(0, device=self.device, dtype=torch.float32)
        returns = torch.zeros(0, device=self.device, dtype=torch.float32)
        timesteps = torch.tensor(0, device=self.device, dtype=torch.long).reshape(1, 1)

        episode_return, episode_length = 0, 0
        for t in range(self.max_ep_len):
            # add padding
            actions = torch.cat([actions, torch.zeros((1, self.act_dim), device=self.device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=self.device)])
            returns = torch.cat([returns, torch.zeros(1, device=self.device)])

            if self.config['model_type'] in ['bc']:
                action = model.get_action(
                    images.to(dtype=torch.float32),
                    missions.to(dtype=torch.long),
                    actions.to(dtype=torch.float32),
                    rewards.to(dtype=torch.float32),
                    returns.to(dtype=torch.float32),
                    timesteps.to(dtype=torch.long),
                )
            elif self.config['model_type'] in ['mgdt']:
                _, ret = model.get_action(
                    images.to(dtype=torch.float32),
                    missions.to(dtype=torch.long),
                    actions.to(dtype=torch.float32),
                    rewards.to(dtype=torch.float32),
                    returns.to(dtype=torch.float32),
                    timesteps.to(dtype=torch.long),
                )
                if self.config['sample_return'] == True:
                    eps = torch.randn(self.config['num_sample_return'], 1).to(ret[1].device)
                    ret_tmp = ret[0] + eps * torch.exp(0.5 * ret[1])
                    ret = ret_tmp.max(0)[0]
                returns[-1] = ret
                action, _ = model.get_action(
                    images.to(dtype=torch.float32),
                    missions.to(dtype=torch.long),
                    actions.to(dtype=torch.float32),
                    rewards.to(dtype=torch.float32),
                    returns.to(dtype=torch.float32),
                    timesteps.to(dtype=torch.long),
                )
            actions[-1] = F.one_hot(action.argmax(), self.act_dim)
            action = action.argmax().detach().cpu().numpy()

            state, reward, done, _ = self.env.step(action)
            if reward > 0: reward = max(0, 1 - 0.9 * ((t + 1) / self.max_ep_len))

            cur_image = torch.from_numpy(state['image']).to(device=self.device).reshape(1, *self.image_dim)
            images = torch.cat([images, cur_image], dim=0)
            missions = torch.cat([missions, torch.from_numpy(mission).to(device=self.device, dtype=torch.long)], dim=0)
            rewards[-1] = reward
            timesteps = torch.cat([timesteps, torch.ones((1, 1), device=self.device, dtype=torch.long) * (t + 1)], dim=1)

            episode_return += reward
            episode_length += 1
            
            if done:
                break

        return episode_return, episode_length

    def evaluate_episode_rtg(self, model, env, target_return=None):
        self.env = gym.make(env, disable_env_checker=True)
        model.eval()
        model.to(device=self.device)

        state = self.env.reset()
        image = state['image']
        mission = state['mission']
        mission, _ = preprocess_texts([mission], self.vocab)

        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"
        images = torch.from_numpy(image).reshape(1, *self.image_dim).to(device=self.device, dtype=torch.float32)
        missions = torch.from_numpy(mission).to(device=self.device, dtype=torch.long)
        actions = torch.zeros((0, self.act_dim), device=self.device, dtype=torch.float32)
        rewards = torch.zeros(0, device=self.device, dtype=torch.float32)

        ep_return = target_return
        target_return = torch.tensor(ep_return, device=self.device, dtype=torch.float32).reshape(1, 1)
        timesteps = torch.tensor(0, device=self.device, dtype=torch.long).reshape(1, 1)

        episode_return, episode_length = 0, 0
        for t in range(self.max_ep_len):
            # add padding
            actions = torch.cat([actions, torch.zeros((1, self.act_dim), device=self.device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=self.device)])

            action = model.get_action(
                images.to(dtype=torch.float32),
                missions.to(dtype=torch.long),
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
            )
            actions[-1] = F.one_hot(action.argmax(), self.act_dim)
            action = action.argmax().detach().cpu().numpy()

            state, reward, done, _ = self.env.step(action)
            if reward > 0: reward = max(0, 1 - 0.9 * ((t + 1) / self.max_ep_len))

            cur_image = torch.from_numpy(state['image']).to(device=self.device).reshape(1, *self.image_dim)
            images = torch.cat([images, cur_image], dim=0)
            missions = torch.cat([missions, torch.from_numpy(mission).to(device=self.device, dtype=torch.long)], dim=0)
            rewards[-1] = reward

            pred_return = target_return[0,-1] - reward
            target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat([timesteps, torch.ones((1, 1), device=self.device, dtype=torch.long) * (t + 1)], dim=1)

            episode_return += reward
            episode_length += 1

            if done:
                break

        return episode_return, episode_length

    def eval_fn(self, target_rew):
        def fn(model):
            returns, lengths = [[] for _ in range(len(self.env_name))], [[] for _ in range(len(self.env_name))]
            successes = [[] for _ in range(len(self.env_name))]
            for _ in range(self.num_eval_episodes):
                with torch.no_grad():
                    if self.model_type in ['dt']:
                        for i, env in enumerate(self.env_name):
                            ret, length = self.evaluate_episode_rtg(model, env, target_return=target_rew)
                            returns[i].append(ret)
                            successes[i].append(1 if ret > 0 else 0)
                            lengths[i].append(length)
                    else:
                        for i, env in enumerate(self.env_name):
                            ret, length = self.evaluate_episode(model, env)
                            returns[i].append(ret)
                            successes[i].append(1 if ret > 0 else 0)
                            lengths[i].append(length)
            log = {}
            for i, env in enumerate(self.env_name):
                log[f'{env}_target_{target_rew}_return_mean'] = np.mean(returns[i])
                log[f'{env}_target_{target_rew}_return_std'] = np.std(returns[i])
                log[f'{env}_target_{target_rew}_successes'] = np.mean(successes[i])
                log[f'{env}_target_{target_rew}_length_mean'] = np.mean(lengths[i])
                log[f'{env}_target_{target_rew}_length_std'] = np.std(lengths[i])
            return log
        return fn

