"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

import numpy as np





class InnerConfig:
    # try to keep the hyper-parameters as close as possible to the values shown in run_dt_atari.py and model_atari.py
    def __init__(self):
        self.obs_C = 4
        self.patch_dim = 84
        self.num_blocks = 1
        self.embed_dim_inner = 128
        self.num_heads_inner = 1
        self.attention_dropout_inner = 0.0
        self.ffn_dropout_inner = 0.0
        self.activation_fn_inner = nn.ReLU
        self.dim_expand_inner = 1
        self.have_position_encoding = False
        self.share_tit_blocks = False


class InnerTransformerBlock(nn.Module):
    def __init__(self, config):
        super(InnerTransformerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim_inner)
        self.attention = nn.MultiheadAttention(
            embed_dim=config.embed_dim_inner,
            num_heads=config.num_heads_inner,
            dropout=config.attention_dropout_inner,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(config.embed_dim_inner)
        self.ffn = nn.Sequential(
            nn.Linear(config.embed_dim_inner, config.dim_expand_inner * config.embed_dim_inner),
            config.activation_fn_inner(),
            nn.Linear(config.dim_expand_inner * config.embed_dim_inner, config.embed_dim_inner),
            nn.Dropout(config.ffn_dropout_inner),
        )

    def forward(self, x):
        x_ln1 = self.ln1(x)
        attn_outputs, attn_weights = self.attention(query=x_ln1, key=x_ln1, value=x_ln1)
        x = x + attn_outputs

        x_ln2 = self.ln2(x)
        ffn_outputs = self.ffn(x_ln2)
        x = x + ffn_outputs
        return x





class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)

class GPTConfig_TIT:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig_TIT):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
        #                              .view(1, 1, config.block_size, config.block_size))
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + 1, config.block_size + 1))
                                     .view(1, 1, config.block_size + 1, config.block_size + 1))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT_TIT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.model_type = config.model_type
        print('self.model_type ==>', self.model_type)

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size + 1, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep+1, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)


        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

        '''
        self.state_encoder = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
                                 nn.Flatten(), nn.Linear(3136, config.n_embd), nn.Tanh())
        '''
        # TIT note: change CNN state_embedding by InnerTransformer state_embedding
        # the key idea of TIT is that processing state with one Transformer, and processing sequential states(+action+return) by another Transformer
        inner_config = InnerConfig()
        self.inner_config = inner_config
        assert inner_config.embed_dim_inner == config.n_embd
        self.inner_blocks = nn.ModuleList([InnerTransformerBlock(inner_config) for _ in range(inner_config.num_blocks)])
        self.obs_patch_embed = nn.Conv2d(
            in_channels=inner_config.obs_C,
            out_channels=inner_config.embed_dim_inner,
            kernel_size=inner_config.patch_dim,
            stride=inner_config.patch_dim,
            bias=False,
        )
        self.class_token_encoding = nn.Parameter(torch.zeros(1, 1, inner_config.embed_dim_inner))
        nn.init.trunc_normal_(self.class_token_encoding, mean=0.0, std=0.02)

        self.ret_emb = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())

        self.action_embeddings = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), nn.Tanh())
        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')

        # specific for TIT
        # AssertionError: parameters {'inner_blocks.0.attention.in_proj_weight', 'class_token_encoding'} were not separated into either decay/no_decay set!
        for i in range(self.inner_config.num_blocks):
            decay.add(f'inner_blocks.{i}.attention.in_proj_weight')
        no_decay.add('class_token_encoding')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def _observation_patch_embedding(self, obs):
        B, context_len_outer, C, H, W = obs.size()
        B = B * context_len_outer  # new_B
        obs = obs.contiguous().view(B, C, H, W)
        obs_patch_embedding = self.obs_patch_embed(obs)  # shape is (new_B, out_C, out_H, out_W),
        # where out_C=embed_dim_inner, out_H*out_W=context_len_inner
        obs_patch_embedding = obs_patch_embedding.view(B, self.inner_config.embed_dim_inner, -1)
        obs_patch_embedding = obs_patch_embedding.transpose(2, 1)  # (new_B, context_len_inner, embed_dim_inner)
        return obs_patch_embedding

    # state, action, and return
    def forward(self, states, actions, targets=None, rtgs=None, timesteps=None):
        # states: (batch, block_size, 4*84*84)
        # actions: (batch, block_size, 1)
        # targets: (batch, block_size, 1)
        # rtgs: (batch, block_size, 1)
        # timesteps: (batch, 1, 1)

        # state_embeddings = self.state_encoder(states.reshape(-1, 4, 84, 84).type(torch.float32).contiguous()) # (batch * block_size, n_embd)
        # state_embeddings = state_embeddings.reshape(states.shape[0], states.shape[1], self.config.n_embd) # (batch, block_size, n_embd)
        # TIT note: change Linear state_embedding by InnerTransformer state_embedding
        # the key idea of TIT is that processing state with one Transformer, and processing sequential states(+action+return) by another Transformer
        if len(states.size()) == 3:  # for training
            # print('TIT =======> 3: states.size()', states.size())  # torch.Size([32, 30, 28224])
            B, context_len_outer, C_H_W = states.size()
            states = states.reshape(B, context_len_outer, 4, 84, 84).type(torch.float32).contiguous()
            B, context_len_outer, C, H, W = states.size()
            # print('TIT =======> B, context_len_outer, C, H, W', B, context_len_outer, C, H, W)  # 32 30 4 84 84
        elif len(states.size()) == 5:  # for sampling from line187@trainer_atari.py
            # print('TIT =======> 5: states.size()', states.size())  # torch.Size([1, 1, 4, 84, 84])
            B, context_len_outer, C, H, W = states.size()
        elif len(states.size()) == 6:  # for sampling from line214@trainer_atari.py
            # print('TIT =======> 6: states.size()', states.size())  # torch.Size([1, K, 1, 4, 84, 84]) where K=[2 ~ 30, i.e., 2 ~ block_size]
            states = states.squeeze(0)  # torch.Size([K, 1, 4, 84, 84])
            states = states.squeeze(1)  # torch.Size([K, 4, 84, 84])
            states = states.unsqueeze(0)  # torch.Size([1, K, 4, 84, 84])
            B, context_len_outer, C, H, W = states.size()

        patch_embeddings = self._observation_patch_embedding(states)
        # print('TIT =======> patch_embeddings.shape', patch_embeddings.shape)  # torch.Size([960, 49, 128]) where 960=32*30
        context_len_inner = patch_embeddings.shape[1]
        inner_tokens = torch.cat([self.class_token_encoding.expand(B * context_len_outer, -1, -1), patch_embeddings], dim=1)
        # print('TIT =======> inner_tokens.shape', inner_tokens.shape)  # torch.Size([960, 50, 128])
        for inner_block in self.inner_blocks:
            inner_tokens = inner_block(inner_tokens)
        temp = inner_tokens.view(B, context_len_outer, context_len_inner + 1, self.config.n_embd)
        # print('TIT =======> temp.shape', temp.shape)  # torch.Size([32, 30, 50, 128])
        state_embeddings = temp[:, :, 0, :]  # 0 means class_tokens, which serve as the input of outer DT
        # print('TIT =======> state_embeddings.shape', state_embeddings.shape)  # torch.Size([32, 30, 128])
        
        if actions is not None and self.model_type == 'reward_conditioned_tit':
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size, n_embd)

            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*3 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::3,:] = rtg_embeddings
            token_embeddings[:,1::3,:] = state_embeddings
            token_embeddings[:,2::3,:] = action_embeddings[:,-states.shape[1] + int(targets is None):,:]
        elif actions is None and self.model_type == 'reward_conditioned_tit': # only happens at very first timestep of evaluation
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))

            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*2, self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::2,:] = rtg_embeddings # really just [:,0,:]
            token_embeddings[:,1::2,:] = state_embeddings # really just [:,1,:]
        elif actions is not None and self.model_type == 'naive_tit':
            action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size, n_embd)

            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*2 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::2,:] = state_embeddings
            token_embeddings[:,1::2,:] = action_embeddings[:,-states.shape[1] + int(targets is None):,:]
        elif actions is None and self.model_type == 'naive_tit': # only happens at very first timestep of evaluation
            token_embeddings = state_embeddings
        else:
            raise NotImplementedError()

        batch_size = states.shape[0]
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0) # batch_size, traj_length, n_embd

        position_embeddings = torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1)) + self.pos_emb[:, :token_embeddings.shape[1], :]

        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if actions is not None and self.model_type == 'reward_conditioned_tit':
            logits = logits[:, 1::3, :] # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'reward_conditioned_tit':
            logits = logits[:, 1:, :]
        elif actions is not None and self.model_type == 'naive_tit':
            logits = logits[:, ::2, :] # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'naive_tit':
            logits = logits # for completeness
        else:
            raise NotImplementedError()

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        return logits, loss
