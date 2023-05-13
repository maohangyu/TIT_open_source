"""
highly based on
https://github.com/kzl/decision-transformer/blob/master/atari/mingpt/model_atari.py
https://github.com/kzl/decision-transformer/blob/master/gym/decision_transformer/models/decision_transformer.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import gym
import babyai
import numpy as np


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config['n_embd'] % config['n_head'] == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config['n_embd'], config['n_embd'])
        self.query = nn.Linear(config['n_embd'], config['n_embd'])
        self.value = nn.Linear(config['n_embd'], config['n_embd'])
        # regularization
        self.attn_drop = nn.Dropout(config['attn_pdrop'])
        self.resid_drop = nn.Dropout(config['resid_pdrop'])

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config['n_ctx'], config['n_ctx'])).view(1, 1, config['n_ctx'], config['n_ctx']))
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        
        # output projection
        self.proj = nn.Linear(config['n_embd'], config['n_embd'])
        self.n_head = config['n_head']

    def forward(self, x, mask):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        ## [ B x n_heads x T x head_dim ]
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        ## [ B x n_heads x T x T ]
        mask = mask.view(B, -1)
        mask = mask[:, None, None, :]
        mask = (1.0 - mask) * -10000.0
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = torch.where(self.bias[:, :, :T, :T].bool(), att, self.masked_bias.to(att.dtype))
        att = att + mask
        att = F.softmax(att, dim=-1)
        self._attn_map = att.clone()
        att = self.attn_drop(att)
        ## [ B x n_heads x T x head_size ]
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        ## [ B x T x embedding_dim ]
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y
    
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config['n_embd'])
        self.ln2 = nn.LayerNorm(config['n_embd'])
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config['n_embd'], config['n_inner']),
            nn.GELU(),
            nn.Linear(config['n_inner'], config['n_embd']),
            nn.Dropout(config['resid_pdrop']),
        )

    def forward(self, inputs_embeds, attention_mask):
        x = inputs_embeds + self.attn(self.ln1(inputs_embeds), attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class DecisionTransformer(nn.Module):
    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """
    def __init__(self, config, **kwargs):
        super(DecisionTransformer, self).__init__()

        self.config = config
        self.length_times = config['length_times']
        self.hidden_size = config['hidden_size']
        assert self.hidden_size == config['n_embd']
        self.max_length = config['K']

        self.env = gym.make(config['env_name'][0])
        self.image_dim = np.prod(self.env.observation_space['image'].shape[:2])
        self.act_dim = self.env.action_space.n
        self.max_ep_len = config['max_ep_len']

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        # self.transformer = GPT2Model(config)
        self.transformer = nn.ModuleList([Block(config) for _ in range(config['n_layer'])])

        self.embed_timestep = nn.Embedding(self.max_ep_len, self.hidden_size)
        self.embed_position = nn.Embedding(self.config['text_max_size'] + self.env.observation_space['image'].shape[-1], self.hidden_size)
        self.embed_return = torch.nn.Linear(1, self.hidden_size)
        self.embed_reward = torch.nn.Linear(1, self.hidden_size)
        self.embed_image = torch.nn.Linear(self.image_dim, self.hidden_size)
        self.embed_mission = torch.nn.Embedding(self.config['text_max_size'], self.hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, self.hidden_size)

        self.embed_ln = nn.LayerNorm(self.hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_image = torch.nn.Linear(self.hidden_size, self.image_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(self.hidden_size, self.act_dim)])
        )
        if self.config['model_type'] in ['mgdt']:
            if self.config['sample_return'] == False:
                self.predict_return = torch.nn.Linear(self.hidden_size, 1)
            else:
                self.predict_return_mu = torch.nn.Linear(self.hidden_size, 1)
                self.predict_return_sigma = torch.nn.Linear(self.hidden_size, 1)
        else:
            self.predict_return = torch.nn.Linear(self.hidden_size, 1)
        self.predict_reward = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, images, missions, mission_masks, actions, rewards, returns_to_go, timesteps, attention_mask=None):

        batch_size, seq_length = images.shape[0], images.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        images = images.reshape(batch_size, seq_length, -1, images.size(-1)).permute([0, 1, 3, 2])
        # embed each modality with a different head
        image_embeddings = self.embed_image(images)
        mission_embeddings = self.embed_mission(missions)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        rewards_embeddings = self.embed_reward(rewards)
        time_embeddings = self.embed_timestep(timesteps)
        

        # time embeddings are treated similar to positional embeddings
        image_embeddings = image_embeddings + time_embeddings.unsqueeze(-2)
        mission_embeddings = mission_embeddings + time_embeddings.unsqueeze(-2)
        state_embeddings = torch.cat([image_embeddings, mission_embeddings], dim=-2)
        image_token_num = image_embeddings.size(2)
        mission_token_num = mission_embeddings.size(2)
        state_token_num = state_embeddings.size(2)
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        rewards_embeddings = rewards_embeddings + time_embeddings
        position_embeddings = torch.cumsum(torch.ones([batch_size, seq_length, state_token_num]).to(state_embeddings.device), dim=2).long() - 1
        position_embeddings = self.embed_position(position_embeddings)
        state_embeddings = state_embeddings + position_embeddings
        
        # when evaluating, mission masks are None
        if mission_masks is not None:
            mission_masks = mission_masks
        else:
            mission_masks = torch.ones_like(missions)


        mask_template = torch.ones_like(returns_to_go)
        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        if self.config['model_type'] in ['dt']:
            # [B, T * N, C]
            stacked_inputs = torch.cat(
                (returns_embeddings.unsqueeze(-2), state_embeddings, action_embeddings.unsqueeze(-2)), dim=2
            ).reshape(batch_size, (2+state_token_num)*seq_length, self.hidden_size)
            stacked_inputs = self.embed_ln(stacked_inputs)
            total_token_num = 2 + state_token_num
            # [B, T * N]
            stacked_token_masks = torch.cat(
                [mask_template, mask_template.repeat(1, 1, image_token_num), mission_masks, mask_template], dim=2
            ).reshape(batch_size, total_token_num*seq_length).to(stacked_inputs.dtype)

        elif self.config['model_type'] in ['bc']:
            stacked_inputs = torch.cat(
                (state_embeddings, action_embeddings.unsqueeze(-2)), dim=2
            ).reshape(batch_size, (1+state_token_num)*seq_length, self.hidden_size)
            stacked_inputs = self.embed_ln(stacked_inputs)
            total_token_num = 1 + state_token_num
            stacked_token_masks = torch.cat(
                [mask_template.repeat(1, 1, image_token_num), mission_masks, mask_template], dim=2
            ).reshape(batch_size, total_token_num*seq_length).to(stacked_inputs.dtype)

        elif self.config['model_type'] in ['mgdt']:
            stacked_inputs = torch.cat(
                (state_embeddings, returns_embeddings.unsqueeze(-2), action_embeddings.unsqueeze(-2), rewards_embeddings.unsqueeze(-2)), dim=2
            ).reshape(batch_size, (3+state_token_num)*seq_length, self.hidden_size)
            stacked_inputs = self.embed_ln(stacked_inputs)
            total_token_num = 3 + state_token_num
            stacked_token_masks = torch.cat(
                [mask_template.repeat(1, 1, image_token_num), mission_masks, mask_template, mask_template, mask_template], dim=2
            ).reshape(batch_size, total_token_num*seq_length).to(stacked_inputs.dtype)

        
        # to make the attention mask fit the stacked inputs, have to stack it as well
        # [B, T * N]
        stacked_attention_mask = torch.stack(
            ([attention_mask for _ in range(total_token_num)]), dim=1
        ).permute(0, 2, 1).reshape(batch_size, total_token_num*seq_length).to(stacked_inputs.dtype)
        stacked_attention_mask = stacked_attention_mask * stacked_token_masks

        # we feed in the input embeddings (not word indices as in NLP) to the model
        x = stacked_inputs
        for block in self.transformer:
            x = block(x, stacked_attention_mask)

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, total_token_num, self.hidden_size).permute(0, 2, 1, 3)

        valid_mission_token_num = mission_masks.sum(-1)
        valid_state_token_idx = valid_mission_token_num + image_token_num - 1
        valid_state_token_idx = valid_state_token_idx[:, None, :, None].repeat([1, 1, 1, x.size(-1)]).long()
        # get predictions
        if self.config['model_type'] in ['dt']:
            x_actions = torch.gather(x, 1, 1+valid_state_token_idx).squeeze(1)
            action_preds = self.predict_action(x_actions)  # predict next action given state
            return None, action_preds, None, None
        elif self.config['model_type'] in ['bc']:
            x_actions = torch.gather(x, 1, valid_state_token_idx).squeeze(1)
            action_preds = self.predict_action(x_actions)  # predict next action given state
            return None, action_preds, None, None
        elif self.config['model_type'] in ['mgdt']:
            x_returns = torch.gather(x, 1, valid_state_token_idx).squeeze(1)
            if self.config['sample_return'] == False:
                return_preds = self.predict_return(x_returns)  # predict next return
            else:
                return_preds_mu = self.predict_return_mu(x_returns)
                return_preds_sigma = self.predict_return_sigma(x_returns)
                # eps = torch.randn_like(return_preds_sigma)
                # return_preds = return_preds_mu + eps * torch.exp(0.5 * return_preds_sigma)
            x_rewards = torch.gather(x, 1, 2+valid_state_token_idx).squeeze(1)
            reward_preds = self.predict_reward(x_rewards)  # predict next rewards
            x_actions = torch.gather(x, 1, 1+valid_state_token_idx).squeeze(1)
            action_preds = self.predict_action(x_actions)  # predict next action
            if self.config['sample_return'] == False:
                return None, action_preds, return_preds, reward_preds
            else:
                return None, action_preds, [return_preds_mu, return_preds_sigma], reward_preds

    def get_action(self, images, missions, actions, rewards, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model
        images = images.reshape(1, -1, *images.size()[-3:])
        missions = missions.reshape(1, -1, missions.size(-1))
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        rewards = rewards.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            images = images[:,-self.max_length:]
            missions = missions[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            rewards = rewards[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-images.shape[1]), torch.ones(images.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=images.device).reshape(1, -1)
            images = torch.cat(
                [torch.zeros((images.shape[0], self.max_length-images.shape[1], *images.size()[-3:]), device=images.device), images],
                dim=1).to(dtype=torch.float32)
            missions = torch.cat(
                [torch.zeros((missions.shape[0], self.max_length-missions.shape[1], missions.size(-1)), device=missions.device), missions],
                dim=1).to(dtype=torch.long)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length-actions.shape[1], self.act_dim), device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            rewards = torch.cat(
                [torch.zeros((rewards.shape[0], self.max_length-rewards.shape[1], 1), device=rewards.device), rewards],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, return_preds, reward_preds = self.forward(
            images, missions, None, actions, rewards, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)

        if self.config['model_type'] in ['bc', 'dt']:
            return action_preds[0, -1]
        elif self.config['model_type'] in ['mgdt']:
            if self.config['sample_return'] == False:
                return action_preds[0, -1], return_preds[0, -1]
            else:
                return action_preds[0, -1], [return_preds[0][0, -1], return_preds[1][0, -1]]


class TIT_DecisionTransformer(nn.Module):
    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """
    def __init__(self, config, **kwargs):
        super(TIT_DecisionTransformer, self).__init__()

        self.config = config
        self.length_times = config['length_times']
        self.hidden_size = config['hidden_size']
        assert self.hidden_size == config['n_embd']
        self.max_length = config['K']

        self.env = gym.make(config['env_name'][0])
        self.image_dim = np.prod(self.env.observation_space['image'].shape[:2])
        self.act_dim = self.env.action_space.n
        self.max_ep_len = config['max_ep_len']

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        # self.transformer = GPT2Model(config)
        self.transformer = nn.ModuleList([Block(config) for _ in range(config['n_layer'])])
        self.inner_transformer = nn.ModuleList([Block(config['inner']) for _ in range(config['inner']['n_layer'])])

        self.embed_timestep = nn.Embedding(self.max_ep_len, self.hidden_size)
        self.inner_embed_timestep = nn.Embedding(self.env.observation_space['image'].shape[-1] + self.config['text_max_size'], self.hidden_size)
        self.embed_return = torch.nn.Linear(1, self.hidden_size)
        self.embed_reward = torch.nn.Linear(1, self.hidden_size)
        self.embed_image = torch.nn.Linear(self.image_dim, self.hidden_size)
        self.embed_mission = torch.nn.Embedding(self.config['text_max_size'], self.hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, self.hidden_size)

        self.embed_ln = nn.LayerNorm(self.hidden_size)
        self.inner_embed_ln = nn.LayerNorm(self.hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_image = torch.nn.Linear(self.hidden_size, self.image_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(self.hidden_size, self.act_dim)])
        )
        if self.config['model_type'] in ['mgdt']:
            if self.config['sample_return'] == False:
                self.predict_return = torch.nn.Linear(self.hidden_size, 1)
            else:
                self.predict_return_mu = torch.nn.Linear(self.hidden_size, 1)
                self.predict_return_sigma = torch.nn.Linear(self.hidden_size, 1)
        else:
            self.predict_return = torch.nn.Linear(self.hidden_size, 1)
        self.predict_reward = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, images, missions, mission_masks, actions, rewards, returns_to_go, timesteps, attention_mask=None):

        batch_size, seq_length = images.shape[0], images.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        images = images.reshape(batch_size, seq_length, -1, images.size(-1)).permute([0, 1, 3, 2])
        # embed each modality with a different head
        image_embeddings = self.embed_image(images)
        mission_embeddings = self.embed_mission(missions)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        rewards_embeddings = self.embed_reward(rewards)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        image_embeddings = image_embeddings
        mission_embeddings = mission_embeddings
        state_embeddings = torch.cat([image_embeddings, mission_embeddings], dim=-2)
        image_token_num = image_embeddings.size(2)
        mission_token_num = mission_embeddings.size(2)
        state_token_num = state_embeddings.size(2)
        inner_position_embeddings = torch.cumsum(torch.ones([batch_size, seq_length, state_token_num]).to(state_embeddings.device), dim=2).long() - 1
        inner_position_embeddings = self.inner_embed_timestep(inner_position_embeddings)
        state_embeddings = state_embeddings + inner_position_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        rewards_embeddings = rewards_embeddings + time_embeddings
        
        if mission_masks is not None:
            mission_masks = mission_masks
        else:
            mission_masks = torch.ones_like(missions)
            
        x = state_embeddings.reshape(batch_size*seq_length, state_token_num, self.hidden_size)
        x = self.inner_embed_ln(x)
        stacked_token_masks = torch.cat(
            [torch.ones_like(image_embeddings)[:, :, :, 0], mission_masks], dim=2
        ).reshape(batch_size*seq_length, state_token_num)
        for block in self.inner_transformer:
            x = block(x, stacked_token_masks)
        x = x.reshape(batch_size, seq_length, state_token_num, self.hidden_size).permute(0, 2, 1, 3)
        feat_idx = torch.cat(
            [torch.ones_like(image_embeddings)[:, :, :, 0], mission_masks], dim=2
        ).permute(0, 2, 1).sum(dim=1, keepdim=True).long() - 1
        feat_idx = feat_idx.unsqueeze(-1).repeat([1, 1, 1, x.size(-1)])
        state_embeddings = torch.gather(x, 1, feat_idx).squeeze()
        state_embeddings = state_embeddings + time_embeddings
        

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        if self.config['model_type'] in ['dt']:
            # [B, T * N, C]
            stacked_inputs = torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
            stacked_inputs = self.embed_ln(stacked_inputs)
            length_times = 3

        elif self.config['model_type'] in ['bc']:
            stacked_inputs = torch.stack(
                (state_embeddings, action_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, 2*seq_length, self.hidden_size)
            stacked_inputs = self.embed_ln(stacked_inputs)
            length_times = 2

        elif self.config['model_type'] in ['mgdt']:
            stacked_inputs = torch.stack(
                (state_embeddings, returns_embeddings, action_embeddings, rewards_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, 4*seq_length, self.hidden_size)
            stacked_inputs = self.embed_ln(stacked_inputs)
            length_times = 4

        # to make the attention mask fit the stacked inputs, have to stack it as well
        # [B, T * N]
        stacked_attention_mask = torch.stack(
            ([attention_mask for _ in range(length_times)]), dim=1
        ).permute(0, 2, 1).reshape(batch_size, length_times*seq_length).to(stacked_inputs.dtype)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        x = stacked_inputs
        for block in self.transformer:
            x = block(x, stacked_attention_mask)

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, self.length_times, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        if self.config['model_type'] in ['dt']:
            action_preds = self.predict_action(x[:,1])  # predict next action given state
            return None, action_preds, None, None
        elif self.config['model_type'] in ['bc']:
            action_preds = self.predict_action(x[:,0])  # predict next action given state
            return None, action_preds, None, None
        elif self.config['model_type'] in ['mgdt']:
            if self.config['sample_return'] == False:
                return_preds = self.predict_return(x[:,0])  # predict next return
            else:
                return_preds_mu = self.predict_return_mu(x[:,0])
                return_preds_sigma = self.predict_return_sigma(x[:,0])
                # eps = torch.randn_like(return_preds_sigma)
                # return_preds = return_preds_mu + eps * torch.exp(0.5 * return_preds_sigma)
            reward_preds = self.predict_reward(x[:,2])  # predict next rewards
            action_preds = self.predict_action(x[:,1])  # predict next action
            if self.config['sample_return'] == False:
                return None, action_preds, return_preds, reward_preds
            else:
                return None, action_preds, [return_preds_mu, return_preds_sigma], reward_preds

    def get_action(self, images, missions, actions, rewards, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model
        images = images.reshape(1, -1, *images.size()[-3:])
        missions = missions.reshape(1, -1, missions.size(-1))
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        rewards = rewards.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            images = images[:,-self.max_length:]
            missions = missions[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            rewards = rewards[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-images.shape[1]), torch.ones(images.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=images.device).reshape(1, -1)
            images = torch.cat(
                [torch.zeros((images.shape[0], self.max_length-images.shape[1], *images.size()[-3:]), device=images.device), images],
                dim=1).to(dtype=torch.float32)
            missions = torch.cat(
                [torch.zeros((missions.shape[0], self.max_length-missions.shape[1], missions.size(-1)), device=missions.device), missions],
                dim=1).to(dtype=torch.long)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length-actions.shape[1], self.act_dim), device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            rewards = torch.cat(
                [torch.zeros((rewards.shape[0], self.max_length-rewards.shape[1], 1), device=rewards.device), rewards],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, return_preds, reward_preds = self.forward(
            images, missions, None, actions, rewards, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)

        if self.config['model_type'] in ['bc', 'dt']:
            return action_preds[0, -1]
        elif self.config['model_type'] in ['mgdt']:
            if self.config['sample_return'] == False:
                return action_preds[0, -1], return_preds[0, -1]
            else:
                return action_preds[0, -1], [return_preds[0][0, -1], return_preds[1][0, -1]]
