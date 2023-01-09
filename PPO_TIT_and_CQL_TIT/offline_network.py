import numpy as np
import torch.nn as nn
from d3rlpy.models.encoders import EncoderFactory
from network import Config, TIT


# https://d3rlpy.readthedocs.io/en/v1.1.1/references/network_architectures.html
# [You can also build your own encoder factory.]
# self-defined CQL: 1. define your own neural network
class MyCustomEncoder(nn.Module):
    def __init__(self, observation_space,
                 algo,
                 patch_dim, num_blocks, features_dim,
                 embed_dim_inner, num_heads_inner, attention_dropout_inner, ffn_dropout_inner,
                 embed_dim_outer, num_heads_outer, attention_dropout_outer, ffn_dropout_outer,
                 activation_fn_inner, activation_fn_outer, activation_fn_other,
                 dim_expand_inner, dim_expand_outer, have_position_encoding, share_tit_blocks):
        super(MyCustomEncoder, self).__init__()
        self.features_dim = features_dim

        C, H, W, D = 0, 0, 0, 0
        print("observation_space:", observation_space)
        if len(observation_space) == 3:  # (4, 84, 84)
            observation_type = 'image'
            C, H, W = observation_space[0], observation_space[1], observation_space[2]
            assert (H % patch_dim == 0) and (W % patch_dim == 0)
            context_len_inner = (H // patch_dim) * (W // patch_dim)
            n_stack = 1  # 4
            context_len_outer = n_stack
        elif len(observation_space) == 1:  # (4,)
            observation_type = 'array'
            D = observation_space[0]
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

    def forward(self, observations):
        # print("observations.shape:", observations.shape)  # torch.Size([32, 100])  torch.Size([32, 1, 84, 84])
        return self.pure_transformer_backbone(observations)

    # THIS IS IMPORTANT!
    def get_feature_size(self):
        return self.features_dim


# self-defined CQL: 2. define your own encoder factory
class MyCustomEncoderFactory(EncoderFactory):
    TYPE = 'custom'  # this is necessary

    def __init__(self,
                 algo,
                 patch_dim, num_blocks, features_dim,
                 embed_dim_inner, num_heads_inner, attention_dropout_inner, ffn_dropout_inner,
                 embed_dim_outer, num_heads_outer, attention_dropout_outer, ffn_dropout_outer,
                 activation_fn_inner, activation_fn_outer, activation_fn_other,
                 dim_expand_inner, dim_expand_outer, have_position_encoding, share_tit_blocks):
        self.algo = algo
        self.patch_dim = patch_dim
        self.num_blocks = num_blocks
        self.features_dim = features_dim
        self.embed_dim_inner = embed_dim_inner
        self.num_heads_inner = num_heads_inner
        self.attention_dropout_inner = attention_dropout_inner
        self.ffn_dropout_inner = ffn_dropout_inner
        self.embed_dim_outer = embed_dim_outer
        self.num_heads_outer = num_heads_outer
        self.attention_dropout_outer = attention_dropout_outer
        self.ffn_dropout_outer = ffn_dropout_outer
        self.activation_fn_inner = activation_fn_inner
        self.activation_fn_outer = activation_fn_outer
        self.activation_fn_other = activation_fn_other
        self.dim_expand_inner = dim_expand_inner
        self.dim_expand_outer = dim_expand_outer
        self.have_position_encoding = have_position_encoding
        self.share_tit_blocks = share_tit_blocks

    def create(self, observation_shape):
        return MyCustomEncoder(
            observation_shape,
            self.algo,
            self.patch_dim, self.num_blocks, self.features_dim,
            self.embed_dim_inner, self.num_heads_inner, self.attention_dropout_inner, self.ffn_dropout_inner,
            self.embed_dim_outer, self.num_heads_outer, self.attention_dropout_outer, self.ffn_dropout_outer,
            self.activation_fn_inner, self.activation_fn_outer, self.activation_fn_other,
            self.dim_expand_inner, self.dim_expand_outer, self.have_position_encoding, self.share_tit_blocks
        )

    def get_params(self, deep=False):
        return {'feature_size': self.features_dim}

