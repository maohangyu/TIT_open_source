import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


class Config:
    def __init__(self,
                 algo,
                 patch_dim,
                 num_blocks,
                 features_dim,
                 embed_dim_inner,
                 num_heads_inner,
                 attention_dropout_inner,
                 ffn_dropout_inner,
                 context_len_inner,
                 embed_dim_outer,
                 num_heads_outer,
                 attention_dropout_outer,
                 ffn_dropout_outer,
                 context_len_outer,
                 observation_type,
                 C, H, W, D,
                 activation_fn_inner,
                 activation_fn_outer,
                 activation_fn_other,
                 dim_expand_inner,
                 dim_expand_outer,
                 have_position_encoding,
                 share_tit_blocks):
        self.algo = algo
        self.patch_dim = patch_dim
        self.num_blocks = num_blocks
        self.features_dim = features_dim
        self.embed_dim_inner = embed_dim_inner
        self.num_heads_inner = num_heads_inner
        self.attention_dropout_inner = attention_dropout_inner
        self.ffn_dropout_inner = ffn_dropout_inner
        self.context_len_inner = context_len_inner
        self.embed_dim_outer = embed_dim_outer
        self.num_heads_outer = num_heads_outer
        self.attention_dropout_outer = attention_dropout_outer
        self.ffn_dropout_outer = ffn_dropout_outer
        self.context_len_outer = context_len_outer
        self.observation_type = observation_type
        self.obs_C, self.obs_H, self.obs_W, self.obs_D = C, H, W, D
        self.obs_C = 1  # if observation has C channels, we think it has 1 channel with context_len_outer==C
        self.activation_fn_inner = activation_fn_inner
        self.activation_fn_outer = activation_fn_outer
        self.activation_fn_other = activation_fn_other
        self.dim_expand_inner = dim_expand_inner
        self.dim_expand_outer = dim_expand_outer
        self.have_position_encoding = have_position_encoding
        self.share_tit_blocks = share_tit_blocks


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


class OuterTransformerBlock(nn.Module):
    def __init__(self, config):
        super(OuterTransformerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim_outer)
        self.attention = nn.MultiheadAttention(
            embed_dim=config.embed_dim_outer,
            num_heads=config.num_heads_outer,
            dropout=config.attention_dropout_outer,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(config.embed_dim_outer)
        self.ffn = nn.Sequential(
            nn.Linear(config.embed_dim_outer, config.dim_expand_outer * config.embed_dim_outer),
            config.activation_fn_outer(),
            nn.Linear(config.dim_expand_outer * config.embed_dim_outer, config.embed_dim_outer),
            nn.Dropout(config.ffn_dropout_outer),
        )

        # Set up causal masking for attention
        ones = torch.ones(config.context_len_outer, config.context_len_outer)
        self.attention_mask = nn.Parameter(torch.triu(ones, diagonal=1), requires_grad=False)
        self.attention_mask[self.attention_mask.bool()] = -float('inf')
        # The mask will look like:
        # [0, -inf, -inf, ..., -inf]
        # [0,    0, -inf, ..., -inf]
        # ...
        # [0,    0,    0, ...,    0]
        # Where 0 means that timestep is allowed to attend. ==>  For a float mask,
        # the mask values will be added to the attention weight.  ==>
        # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        # So the first timestep can attend only to the first timestep
        # and the last timestep can attend to all observations.

    def forward(self, x):
        x_ln1 = self.ln1(x)
        attn_outputs, attn_weights = self.attention(query=x_ln1, key=x_ln1, value=x_ln1,
                                                    attn_mask=self.attention_mask[:x.size(1), :x.size(1)])
        x = x + attn_outputs

        x_ln2 = self.ln2(x)
        ffn_outputs = self.ffn(x_ln2)
        x = x + ffn_outputs
        return x


class EnhancedBlock(nn.Module):
    def __init__(self, config):
        super(EnhancedBlock, self).__init__()
        self.inner_Transformer_block = InnerTransformerBlock(config)
        self.outer_Transformer_block = OuterTransformerBlock(config)

        self.K = config.context_len_outer
        self.context_len_inner = config.context_len_inner
        self.embed_dim_inner = config.embed_dim_inner

    def forward(self, inner_tokens):
        # inner_tokens has a shape of (new_B, context_len_inner+1, embed_dim_inner) where new_B = B * context_len_outer
        inner_outputs = self.inner_Transformer_block(inner_tokens)

        # outer_tokens has a shape of (B, context_len_outer, embed_dim_outer)
        # for TIT, embed_dim_outer==embed_dim_inner
        temp = inner_outputs.view(-1, self.K, self.context_len_inner+1, self.embed_dim_inner)  # -1 -> B
        outer_tokens = temp[:, :, 0, :]  # 0 means class_tokens, which serve as the input of outer block
        outer_outputs = self.outer_Transformer_block(outer_tokens)

        return inner_outputs, outer_outputs


class TIT(nn.Module):
    def __init__(self, config):
        super(TIT, self).__init__()
        self.config = config

        # Input
        if config.observation_type == 'image':
            # We map each observation patch into the observation patch embedding with a trainable linear projection
            self.obs_patch_embed = nn.Conv2d(
                in_channels=config.obs_C,
                out_channels=config.embed_dim_inner,
                kernel_size=config.patch_dim,
                stride=config.patch_dim,
                bias=False,
            )
        elif config.observation_type == 'array':
            self.obs_patch_embed = nn.Conv1d(
                in_channels=1,
                out_channels=config.embed_dim_inner,
                kernel_size=config.patch_dim,
                stride=config.patch_dim,
                bias=False,
                padding=int(np.ceil((config.context_len_inner * config.patch_dim - config.obs_D) / 2))
            )
        else:
            raise ValueError('observation must be an 3d-image or 1d-array')

        # The patch position encoding is a trainable parameter
        self.class_token_encoding = nn.Parameter(torch.zeros(1, 1, config.embed_dim_inner))
        if self.config.have_position_encoding:
            self.obs_patch_pos_encoding = nn.Parameter(torch.zeros(1, config.context_len_inner+1, config.embed_dim_inner))

        # TiT blocks
        if config.algo in ['vanilla_tit_ppo', 'vanilla_tit_cql']:
            self.inner_blocks = nn.ModuleList([InnerTransformerBlock(config) for _ in range(config.num_blocks)])
            self.outer_blocks = nn.ModuleList([OuterTransformerBlock(config) for _ in range(config.num_blocks)])
        elif config.algo in ['enhanced_tit_ppo', 'enhanced_tit_cql']:
            if self.config.share_tit_blocks:
                self.block = EnhancedBlock(config)  # share parameters between layers
            else:
                self.blocks = nn.ModuleList([EnhancedBlock(config) for _ in range(config.num_blocks)])
            # self.ln1s = nn.ModuleList([nn.LayerNorm(config.embed_dim_outer) for _ in range(config.num_blocks)])
        else:
            raise ValueError('model_type must be Vanilla_TIT, Fused_TIT or Enhanced_TIT')

        # Head
        if config.algo in ['vanilla_tit_ppo', 'vanilla_tit_cql']:
            self.ln1 = nn.LayerNorm(config.embed_dim_outer)
            self.head = nn.Sequential(
                nn.Linear(config.embed_dim_outer, config.features_dim),
                config.activation_fn_other()
            )
            self.ln2 = nn.LayerNorm(config.features_dim)
        elif config.algo in ['enhanced_tit_ppo', 'enhanced_tit_cql']:
            self.ln1 = nn.LayerNorm(config.embed_dim_outer * config.num_blocks)
            self.head = nn.Sequential(
                nn.Linear(config.embed_dim_outer * config.num_blocks, config.features_dim),
                config.activation_fn_other()
            )
            self.ln2 = nn.LayerNorm(config.features_dim)

        nn.init.trunc_normal_(self.class_token_encoding, mean=0.0, std=0.02)
        if self.config.have_position_encoding:
            nn.init.trunc_normal_(self.obs_patch_pos_encoding, mean=0.0, std=0.02)
        self.apply(self._init_weights)

    def _image_observation_patch_embedding(self, obs):
        B, context_len_outer, C, H, W = obs.size()
        B = B * context_len_outer  # new_B
        obs = obs.contiguous().view(B, C, H, W)
        obs_patch_embedding = self.obs_patch_embed(obs)  # shape is (new_B, out_C, out_H, out_W),
        # where out_C=embed_dim_inner, out_H*out_W=context_len_inner
        obs_patch_embedding = obs_patch_embedding.view(B, self.config.embed_dim_inner, self.config.context_len_inner)
        obs_patch_embedding = obs_patch_embedding.transpose(2, 1)  # (new_B, context_len_inner, embed_dim_inner)
        return obs_patch_embedding

    def _array_observation_patch_embedding(self, obs):
        B, context_len_outer, D = obs.size()
        B = B * context_len_outer  # new_B
        obs = obs.view(B, D)
        obs = torch.unsqueeze(obs, dim=1)  # (new_B, 1, D), first apply unsqueeze() before applying Conv1d()
        obs_patch_embedding = self.obs_patch_embed(obs)  # shape is (new_B, out_C, out_length),
        # where out_C=embed_dim_inner, out_length=context_len_inner
        obs_patch_embedding = obs_patch_embedding.transpose(2, 1)  # (new_B, context_len_inner, embed_dim_inner)
        return obs_patch_embedding

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, obs):
        if self.config.observation_type == 'image':
            # print('obs.size() ==>', obs.size())  # we use make_atari_env(n_envs=8), so the size is  (8, 4, 84, 84)
            obs = obs.unsqueeze(dim=2)
            # print('obs.size() ==>', obs.size())  # (8, 4, 1, 84, 84)  4_frames_stack means the context_len_outer=4
            B, context_len_outer, C, H, W = obs.size()
            new_B = B * context_len_outer

            obs_patch_embedding = self._image_observation_patch_embedding(obs)
            inner_tokens = torch.cat([self.class_token_encoding.expand(new_B, -1, -1), obs_patch_embedding], dim=1)
            if self.config.have_position_encoding:
                inner_tokens = inner_tokens + self.obs_patch_pos_encoding
            # inner_tokens has a shape of (new_B, context_len_inner+1, embed_dim_inner)

        elif self.config.observation_type == 'array':
            # print('obs.size() ==>', obs.size())  # (1, 4)
            obs = obs.unsqueeze(dim=1)
            # print('obs.size() ==>', obs.size())  # (1, 1, 4)
            B, context_len_outer, D = obs.size()
            new_B = B * context_len_outer

            obs_patch_embedding = self._array_observation_patch_embedding(obs)
            inner_tokens = torch.cat([self.class_token_encoding.expand(new_B, -1, -1), obs_patch_embedding], dim=1)
            if self.config.have_position_encoding:
                inner_tokens = inner_tokens + self.obs_patch_pos_encoding
            # inner_tokens has a shape of (new_B, context_len_inner+1, embed_dim_inner)

        # inner_tokens has a shape of (new_B, context_len_inner+1, embed_dim_inner)
        # outer_tokens has a shape of (B, context_len_outer, embed_dim_outer)

        if self.config.algo in ['vanilla_tit_ppo', 'vanilla_tit_cql']:
            for inner_block in self.inner_blocks:
                inner_tokens = inner_block(inner_tokens)

            temp = inner_tokens.view(B, context_len_outer, self.config.context_len_inner+1, self.config.embed_dim_inner)
            outer_tokens = temp[:, :, 0, :]  # 0 means class_tokens, which serve as the input of outer block
            for outer_block in self.outer_blocks:
                outer_tokens = outer_block(outer_tokens)

            x = outer_tokens[:, -1, :]  # only return the last element of outer_block for decision-making
            x = self.ln2(self.head(self.ln1(x)))  # (B, embed_dim_outer)

        elif self.config.algo in ['enhanced_tit_ppo', 'enhanced_tit_cql']:
            # if self.config.observation_type == 'array' and self.config.patch_dim > 1:
            #     all_outer_outputs = [inner_tokens[:, -1, :]]
            # else:
            all_outer_outputs = []
            if self.config.share_tit_blocks:
                for i in range(self.config.num_blocks):
                    inner_tokens, outer_outputs = self.block(inner_tokens)
                    all_outer_outputs.append(outer_outputs[:, -1, :])
            else:
                for block in self.blocks:
                    inner_tokens, outer_outputs = block(inner_tokens)
                    all_outer_outputs.append(outer_outputs[:, -1, :])

            x = torch.cat(all_outer_outputs, dim=-1)
            x = self.ln2(self.head(self.ln1(x)))  # (B, embed_dim_outer)

        return x


class TitFeaturesExtractor(BaseFeaturesExtractor):
    # https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#custom-feature-extractor
    # features_dim: Number of features extracted. This corresponds to the number of unit for the last layer.
    def __init__(self, observation_space,
                 algo,
                 patch_dim, num_blocks, features_dim,
                 embed_dim_inner, num_heads_inner, attention_dropout_inner, ffn_dropout_inner,
                 embed_dim_outer, num_heads_outer, attention_dropout_outer, ffn_dropout_outer,
                 activation_fn_inner, activation_fn_outer, activation_fn_other,
                 dim_expand_inner, dim_expand_outer, have_position_encoding, share_tit_blocks):
        super(TitFeaturesExtractor, self).__init__(observation_space, features_dim)

        C, H, W, D = 0, 0, 0, 0
        if len(observation_space.shape) == 3:  # (4, 84, 84)
            observation_type = 'image'
            C, H, W = observation_space.shape
            assert (H % patch_dim == 0) and (W % patch_dim == 0)
            context_len_inner = (H // patch_dim) * (W // patch_dim)
            n_stack = 4
            context_len_outer = n_stack
        elif len(observation_space.shape) == 1:  # (4,)
            observation_type = 'array'
            D = observation_space.shape[0]
            # patch_dim = 1
            # assert patch_dim == 1
            # context_len_inner = D // patch_dim
            context_len_inner = int(np.ceil(D / patch_dim))
            n_stack = 1
            context_len_outer = n_stack
        else:
            raise ValueError('len(observation_space.shape) should either be 1 or 3')
        config = Config(algo,
                        patch_dim,
                        num_blocks,
                        features_dim,
                        embed_dim_inner,
                        num_heads_inner,
                        attention_dropout_inner,
                        ffn_dropout_inner,
                        context_len_inner,
                        embed_dim_outer,
                        num_heads_outer,
                        attention_dropout_outer,
                        ffn_dropout_outer,
                        context_len_outer,
                        observation_type,
                        C, H, W, D,
                        activation_fn_inner,
                        activation_fn_outer,
                        activation_fn_other,
                        dim_expand_inner,
                        dim_expand_outer,
                        have_position_encoding,
                        share_tit_blocks)
        self.pure_transformer_backbone = TIT(config)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.pure_transformer_backbone(observations)





class OFENet(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(self, feature_dim, hidden_dim=64):
        super(OFENet, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = feature_dim + hidden_dim * 2
        self.latent_dim_vf = feature_dim + hidden_dim * 2

        # Policy network
        self.linear1_policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.Tanh())  # use default Tanh as ppo
        self.linear2_policy = nn.Sequential(nn.Linear(feature_dim + hidden_dim, hidden_dim), nn.Tanh())

        # Value network
        self.linear1_value = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.Tanh())  # use default Tanh as ppo
        self.linear2_value = nn.Sequential(nn.Linear(feature_dim + hidden_dim, hidden_dim), nn.Tanh())

    def forward(self, features: torch.Tensor):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        h1 = self.linear1_policy(features)
        h1 = torch.cat([features, h1], dim=-1)

        h2 = self.linear2_policy(h1)
        h2 = torch.cat([h1, h2], dim=-1)

        return h2

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        h1 = self.linear1_value(features)
        h1 = torch.cat([features, h1], dim=-1)

        h2 = self.linear2_value(h1)
        h2 = torch.cat([h1, h2], dim=-1)

        return h2


class OFENetActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, activation_fn=nn.Tanh,
                 *args, **kwargs):
        super(OFENetActorCriticPolicy, self).__init__(observation_space, action_space, lr_schedule, net_arch,
                                                      activation_fn, *args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = OFENet(self.features_dim)


# https://github.com/takuseno/d3rlpy/blob/master/d3rlpy/models/encoders.py#L327
# https://github.com/takuseno/d3rlpy/blob/cd7681ca150d89422f9865daaaa896ead13a7b73/d3rlpy/models/torch/encoders.py#L268
# We check that our D2RLNet is implemented correctly by comparing forward_actor with the above d3rlpy's implementation.
class D2RLNet(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(self, feature_dim, hidden_dim=64):
        super(D2RLNet, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = feature_dim + hidden_dim  # NOTE: this is the only difference between OFENet and D2RLNet
        self.latent_dim_vf = feature_dim + hidden_dim  # NOTE: this is the only difference between OFENet and D2RLNet

        # Policy network
        self.linear1_policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.Tanh())  # use default Tanh as ppo
        self.linear2_policy = nn.Sequential(nn.Linear(feature_dim + hidden_dim, hidden_dim), nn.Tanh())

        # Value network
        self.linear1_value = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.Tanh())  # use default Tanh as ppo
        self.linear2_value = nn.Sequential(nn.Linear(feature_dim + hidden_dim, hidden_dim), nn.Tanh())

    def forward(self, features: torch.Tensor):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        h1 = self.linear1_policy(features)
        h1 = torch.cat([features, h1], dim=-1)

        h2 = self.linear2_policy(h1)
        h2 = torch.cat([features, h2], dim=-1)  # NOTE: this is the only difference between OFENet and D2RLNet

        return h2

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        h1 = self.linear1_value(features)
        h1 = torch.cat([features, h1], dim=-1)

        h2 = self.linear2_value(h1)
        h2 = torch.cat([features, h2], dim=-1)  # NOTE: this is the only difference between OFENet and D2RLNet

        return h2


class D2RLNetActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, activation_fn=nn.Tanh,
                 *args, **kwargs):
        super(D2RLNetActorCriticPolicy, self).__init__(observation_space, action_space, lr_schedule, net_arch,
                                                      activation_fn, *args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = D2RLNet(self.features_dim)





class ResidualBlock(nn.Module):
    def __init__(self, channel):
        super(ResidualBlock, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return x + self.cnn(x)


# https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/torch_layers.py#L51  ==> NatureCNN
# We try to match the structure of Resnet with NatureCNN
class ResnetFeaturesExtractor(BaseFeaturesExtractor):
    # https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#custom-feature-extractor
    # features_dim: Number of features extracted. This corresponds to the number of unit for the last layer.
    def __init__(self, observation_space, features_dim=512):
        super(ResnetFeaturesExtractor, self).__init__(observation_space, features_dim)

        C, H, W = observation_space.shape
        res_channel = 64
        self.resnet = nn.Sequential(
            nn.Conv2d(in_channels=C, out_channels=res_channel, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            ResidualBlock(channel=res_channel),
            ResidualBlock(channel=res_channel),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.resnet(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.resnet(observations))





class CatformerAttention(nn.Module):
    def __init__(self, index_of_layer=0, embed_dim=255, num_heads=5, attention_dropout=0.0, context_len=4):
        super(CatformerAttention, self).__init__()

        self.index_of_layer = index_of_layer
        if self.index_of_layer > 0:
            self.proj = nn.Linear(index_of_layer * embed_dim, embed_dim)

        self.ln = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )

        # Set up causal masking for attention
        ones = torch.ones(context_len, context_len)
        self.attention_mask = nn.Parameter(torch.triu(ones, diagonal=1), requires_grad=False)
        self.attention_mask[self.attention_mask.bool()] = -float('inf')
        # The mask will look like:
        # [0, -inf, -inf, ..., -inf]
        # [0,    0, -inf, ..., -inf]
        # ...
        # [0,    0,    0, ...,    0]
        # Where 0 means that timestep is allowed to attend. ==>  For a float mask,
        # the mask values will be added to the attention weight.  ==>
        # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        # So the first timestep can attend only to the first timestep
        # and the last timestep can attend to all observations.

    def forward(self, x):
        # we do not take the [dimension expansion operation] inside the self.attention Layer, but outside the layer
        if self.index_of_layer > 0:
            x = self.proj(x)

        x_ln = self.ln(x)
        attn_outputs, attn_weights = self.attention(query=x_ln, key=x_ln, value=x_ln,
                                                    attn_mask=self.attention_mask[:x.size(1), :x.size(1)])
        return attn_outputs


class CatformerFFN(nn.Module):
    def __init__(self, index_of_layer=1, embed_dim=255, ffn_dropout=0.0):
        super(CatformerFFN, self).__init__()
        self.ln = nn.LayerNorm(index_of_layer * embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(index_of_layer * embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(ffn_dropout),
        )

    def forward(self, x):
        x_ln = self.ln(x)
        ffn_outputs = self.ffn(x_ln)
        return ffn_outputs


class Catformer(nn.Module):
    """
    in the original Catformer paper,
    Section4.2-[Network size] and AppendixD.1-[Architecture details: controlling parameters] are quite complex, the
    authors only mentioned "We use a = 2; e = 4 for all experiments in Sections 5.1 and 5.2.",
    but we still don't know how to set the expansion factor for RL tasks (i.e., Section 5.3),

    besides, AppendixE.2-[DMLab30] mentioned
    "Our transformer agents have torso MLP sizes of 256, 5 heads, and 2 transformer blocks."
    but it is confusing that a standard Transformer should have embedding_size%head_count==0,

    so we finally try to:
    1) make the network-structure as close to Appendex-[Figure 4] as possible
    2) and do not take the [dimension expansion operation] inside the self.attention Layer, but outside the layer
    3) use the following hyperparameters: embed_dim=255, num_heads=5 to make sure embedding_size%head_count==0
    """
    def __init__(self, n_flatten, embed_dim=255):
        super(Catformer, self).__init__()
        self.embed_dim = embed_dim
        self.context_len = 4  # the same as frame_stack
        self.num_blocks = 2
        self.num_heads = 5

        # input
        self.obs_embedding = nn.Linear(n_flatten, self.embed_dim)
        self.obs_pos_encoding = nn.Parameter(torch.zeros(1, self.context_len, self.embed_dim))

        # block1
        self.layer0_attention_layer = CatformerAttention(index_of_layer=0)
        self.layer1_ffn_layer = CatformerFFN(index_of_layer=1)

        # block2
        self.layer2_attention_layer = CatformerAttention(index_of_layer=2)
        self.layer3_ffn_layer = CatformerFFN(index_of_layer=3)

        nn.init.trunc_normal_(self.obs_pos_encoding, mean=0.0, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, obs):
        # input
        obs_embedding = self.obs_embedding(obs)  # (B, context_len, embed_dim)
        input_tokens = obs_embedding + self.obs_pos_encoding  # (B, context_len, embed_dim)

        # block1
        layer0_attention_layer_outputs = self.layer0_attention_layer(input_tokens)  # (B, context_len, embed_dim)
        layer1_ffn_layer_outputs = self.layer1_ffn_layer(layer0_attention_layer_outputs)  # (B, context_len, embed_dim)

        # block2
        # note the following concatenation, which is Catformer's specific design!
        concat_before_layer2 = torch.cat([layer0_attention_layer_outputs, layer1_ffn_layer_outputs], dim=-1)
        layer2_attention_layer_outputs = self.layer2_attention_layer(concat_before_layer2)
        concat_before_layer3 = torch.cat([concat_before_layer2, layer2_attention_layer_outputs], dim=-1)
        layer3_ffn_layer_outputs = self.layer3_ffn_layer(concat_before_layer3)  # (B, context_len, embed_dim)

        return layer3_ffn_layer_outputs[:, -1, :]


class CatformerFeaturesExtractor(BaseFeaturesExtractor):
    # https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#custom-feature-extractor
    # features_dim: Number of features extracted. This corresponds to the number of unit for the last layer.
    def __init__(self, observation_space, features_dim=512):
        super(CatformerFeaturesExtractor, self).__init__(observation_space, features_dim)

        C, res_channel = 1, 64
        self.resnet = nn.Sequential(
            nn.Conv2d(in_channels=C, out_channels=res_channel, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            ResidualBlock(channel=res_channel),
            ResidualBlock(channel=res_channel),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with torch.no_grad():
            self.n_flatten = self.resnet(torch.as_tensor(observation_space.sample()[None][:,0:1,:,:]).float()).shape[1]

        embed_dim = 255
        self.catformer = Catformer(self.n_flatten, embed_dim=embed_dim)

        self.ln1 = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(nn.Linear(embed_dim, features_dim), nn.ReLU())
        self.ln2 = nn.LayerNorm(features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # print('observations.size() ==>', observations.size())
        # because we use make_atari_env(n_envs=8), so the size is  (8, 4, 84, 84)
        observations = observations.unsqueeze(dim=2)  # (8, 4, 1, 84, 84)  4_frames_stack means the context_len=4
        B, context_len, C, H, W = observations.size()
        new_B = B * context_len
        observations = observations.contiguous().view(new_B, C, H, W)  # (8*4, 1, 84, 84)

        obs_tokens = self.resnet(observations)  # (8*4, n_flatten)

        obs_tokens = obs_tokens.view(B, context_len, self.n_flatten)  # (8, 4, n_flatten)
        catformer_outputs = self.catformer(obs_tokens)  # (8, embed_dim)

        outputs = self.ln2(self.head(self.ln1(catformer_outputs)))  # (B, features_dim)
        return outputs

