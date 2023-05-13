"""
highly based on
https://github.com/kzl/decision-transformer/blob/master/gym/experiment.py#L166
https://github.com/kzl/decision-transformer/blob/master/gym/decision_transformer/evaluation/evaluate_episodes.py
"""

import numpy as np
import torch
import gym


class Evaluation(object):
    def __init__(self, config, state_mean, state_std):
        self.config = config
        self.state_mean = state_mean
        self.state_std = state_std
        self.device = config.get('device', 'cuda')
        self.env_name = config['env_name']
        self.max_ep_len = config['max_ep_len']
        self.scale = config['scale']
        self.num_eval_episodes = config['num_eval_episodes']
        self.model_type = config['model_type']
        
        self.env = gym.make(self.env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        
        self.is_delayed_reward = config['delayed_reward']
        
    def evaluate_episode(self, model, target_return=None):
        model.eval()
        model.to(device=self.device)

        state_mean = torch.from_numpy(self.state_mean).to(device=self.device)
        state_std = torch.from_numpy(self.state_std).to(device=self.device)

        state = self.env.reset()

        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"
        states = torch.from_numpy(state).reshape(1, self.state_dim).to(device=self.device, dtype=torch.float32)
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
                    (states.to(dtype=torch.float32) - state_mean) / state_std,
                    actions.to(dtype=torch.float32),
                    rewards.to(dtype=torch.float32),
                    returns.to(dtype=torch.float32),
                    timesteps.to(dtype=torch.long),
                )
            elif self.config['model_type'] in ['mgdt']:
                _, ret = model.get_action(
                    (states.to(dtype=torch.float32) - state_mean) / state_std,
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
                    (states.to(dtype=torch.float32) - state_mean) / state_std,
                    actions.to(dtype=torch.float32),
                    rewards.to(dtype=torch.float32),
                    returns.to(dtype=torch.float32),
                    timesteps.to(dtype=torch.long),
                )
            actions[-1] = action
            action = action.detach().cpu().numpy()

            state, reward, done, _ = self.env.step(action)

            cur_state = torch.from_numpy(state).to(device=self.device).reshape(1, self.state_dim)
            states = torch.cat([states, cur_state], dim=0)
            if self.config['model_type'] in ['mgdt']:  # only MGDT actually uses the reward, so we should normalize it
                rewards[-1] = -1 + 2 * (reward - self.config['reward_scale'][0]) / (self.config['reward_scale'][1] - self.config['reward_scale'][0])
            elif self.config['model_type'] in ['bc']:
                rewards[-1] = reward
            timesteps = torch.cat([timesteps, torch.ones((1, 1), device=self.device, dtype=torch.long) * (t + 1)], dim=1)

            episode_return += reward
            episode_length += 1

            if done:
                break

        return episode_return, episode_length

    def evaluate_episode_rtg(self, model, target_return=None):
        model.eval()
        model.to(device=self.device)

        state_mean = torch.from_numpy(self.state_mean).to(device=self.device)
        state_std = torch.from_numpy(self.state_std).to(device=self.device)

        state = self.env.reset()

        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"
        states = torch.from_numpy(state).reshape(1, self.state_dim).to(device=self.device, dtype=torch.float32)
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
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
            )
            actions[-1] = action
            action = action.detach().cpu().numpy()

            state, reward, done, _ = self.env.step(action)

            cur_state = torch.from_numpy(state).to(device=self.device).reshape(1, self.state_dim)
            states = torch.cat([states, cur_state], dim=0)
            rewards[-1] = reward

            if self.is_delayed_reward != 'delayed':
                pred_return = target_return[0,-1] - (reward / self.scale)
            else:
                pred_return = target_return[0,-1]
            target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat([timesteps, torch.ones((1, 1), device=self.device, dtype=torch.long) * (t + 1)], dim=1)

            episode_return += reward
            episode_length += 1

            if done:
                break

        return episode_return, episode_length

    def eval_fn(self, target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in range(self.num_eval_episodes):
                with torch.no_grad():
                    if self.model_type in ['dt']:
                        ret, length = self.evaluate_episode_rtg(model, target_return=target_rew/self.scale)
                    else:
                        ret, length = self.evaluate_episode(model, target_return=target_rew/self.scale)
                returns.append(ret)
                lengths.append(length)
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
            }
        return fn

