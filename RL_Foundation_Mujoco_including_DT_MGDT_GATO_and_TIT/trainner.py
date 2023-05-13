"""
highly based on https://github.com/kzl/decision-transformer/blob/master/gym/decision_transformer/training/seq_trainer.py
"""

import numpy as np
import torch
# import wandb

import time
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm


class Trainer:
    def __init__(self, model, optimizer, batch_size, dataset, writer, config, scheduler=None, eval_fns=None):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.dataset = dataset
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.writer = writer
        self.model_type = config['model_type']
        self.reward_scale = config['reward_scale']
        self.config = config
        
        self.train_count = 0

        self.start_time = time.time()

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):
        train_losses = []
        logs = dict()

        train_start = time.time()
        sampler = WeightedRandomSampler(self.dataset.p_sample, num_samples=num_steps*self.batch_size, replacement=True)
        dataloader = DataLoader(self.dataset, sampler=sampler, batch_size=self.batch_size)

        self.model.train()
        for states, actions, rewards, dones, rtg, timesteps, attention_mask in tqdm(dataloader):
            train_loss = self.train_step(states, actions, rewards, dones, rtg, timesteps, attention_mask)
            train_losses.append(train_loss)
            if self.writer is not None:
                self.writer.add_scalar('train_loss', train_loss, self.train_count)
            self.train_count += 1
            if self.scheduler is not None:
                self.scheduler.step()

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        self.model.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    def train_step(self, states, actions, rewards, dones, rtg, timesteps, attention_mask):
        rewards_target, action_target, rtg_target = torch.clone(rewards), torch.clone(actions), torch.clone(rtg)

        state_preds, action_preds, return_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        if self.model_type in ['dt', 'bc']:
            loss = torch.mean((action_preds - action_target) ** 2)
        elif self.model_type in ['mgdt']:
            if self.config['sample_return'] == True:
                eps = torch.randn_like(return_preds[1])
                return_preds_tmp = return_preds[0] + eps * torch.exp(0.5 * return_preds[1])
                return_preds = return_preds_tmp
            return_preds = return_preds.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
            return_target = rtg_target[:,:-1].reshape(-1, 1)[attention_mask.reshape(-1) > 0]
            reward_preds = reward_preds.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
            reward_target = rewards_target.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
            loss = torch.mean((action_preds - action_target) ** 2) \
                + torch.mean((return_preds - return_target) ** 2) \
                + torch.mean((reward_preds - reward_target) ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()

