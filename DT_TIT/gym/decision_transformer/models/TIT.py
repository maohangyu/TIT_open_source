import numpy as np
import torch
import torch.nn as nn

import transformers

from decision_transformer.models.model import TrajectoryModel
from decision_transformer.models.trajectory_gpt2 import GPT2Model


class InnerConfig:
    # try to keep the hyper-parameters as close as possible to the values shown in experiment.py
    def __init__(self):
        self.patch_dim = 11
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


class TIT(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        # self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
        # TIT note: change Linear state_embedding by InnerTransformer state_embedding
        # the key idea of TIT is that processing state with one Transformer, and processing sequential states(+action+return) by another Transformer
        inner_config = InnerConfig()
        inner_config.patch_dim = state_dim
        assert inner_config.embed_dim_inner == self.hidden_size
        print('inner_config.patch_dim ==>', inner_config.patch_dim, state_dim, self.hidden_size)
        self.inner_blocks = nn.ModuleList([InnerTransformerBlock(inner_config) for _ in range(inner_config.num_blocks)])
        self.obs_patch_embed = nn.Conv1d(
            in_channels=1,
            out_channels=inner_config.embed_dim_inner,
            kernel_size=inner_config.patch_dim,
            stride=inner_config.patch_dim,
            bias=False,
        )
        self.class_token_encoding = nn.Parameter(torch.zeros(1, 1, inner_config.embed_dim_inner))
        nn.init.trunc_normal_(self.class_token_encoding, mean=0.0, std=0.02)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = torch.nn.Linear(hidden_size, 1)

    def _observation_patch_embedding(self, obs):
        B, context_len_outer, D = obs.size()
        B = B * context_len_outer  # new_B
        obs = obs.view(B, D)
        obs = torch.unsqueeze(obs, dim=1)  # (new_B, 1, D), first apply unsqueeze() before applying Conv1d()
        obs_patch_embedding = self.obs_patch_embed(obs)  # shape is (new_B, out_C, out_length),
        # where out_C=embed_dim_inner, out_length=context_len_inner
        obs_patch_embedding = obs_patch_embedding.transpose(2, 1)  # (new_B, context_len_inner, embed_dim_inner)
        return obs_patch_embedding

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):
        # print('TIT =======> states.shape', states.shape)  # torch.Size([64, 20, 11])

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        # state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)
        # TIT note: change Linear state_embedding by InnerTransformer state_embedding
        # the key idea of TIT is that processing state with one Transformer, and processing sequential states(+action+return) by another Transformer
        patch_embeddings = self._observation_patch_embedding(states)
        # print('TIT =======> patch_embeddings.shape', patch_embeddings.shape)  # torch.Size([1280, 11, 128]) where 1280=64*20
        context_len_inner = patch_embeddings.shape[1]
        inner_tokens = torch.cat([self.class_token_encoding.expand(batch_size*seq_length, -1, -1), patch_embeddings], dim=1)
        # print('TIT =======> inner_tokens.shape', inner_tokens.shape)  # torch.Size([1280, 12, 128])
        for inner_block in self.inner_blocks:
            inner_tokens = inner_block(inner_tokens)
        temp = inner_tokens.view(batch_size, seq_length, context_len_inner + 1, self.hidden_size)
        # print('TIT =======> temp.shape', temp.shape)  # torch.Size([64, 20, 12, 128])
        state_embeddings = temp[:, :, 0, :]  # 0 means class_tokens, which serve as the input of outer DT
        # print('TIT =======> state_embeddings.shape', state_embeddings.shape)  # torch.Size([64, 20, 128])

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_return(x[:,2])  # predict next return given state and action
        state_preds = self.predict_state(x[:,2])    # predict next state given state and action
        action_preds = self.predict_action(x[:,1])  # predict next action given state

        return state_preds, action_preds, return_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, return_preds = self.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)

        return action_preds[0,-1]
