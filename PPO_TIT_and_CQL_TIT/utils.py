import os
import json
import torch.nn as nn
import yaml
from network import TitFeaturesExtractor, ResnetFeaturesExtractor, CatformerFeaturesExtractor


def linear_schedule(initial_value: float):
    # https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#learning-rate-schedule
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def update_args(args, hyperparams=None):
    if hyperparams is None:
        yaml_file = './hyperparameter_final_' + args.algo + '.yaml'
        with open(yaml_file) as f:
            hyperparams_dict = yaml.safe_load(f)
            if args.env_name in list(hyperparams_dict.keys()):
                hyperparams = hyperparams_dict[args.env_name]
            else:
                raise ValueError(f'Hyperparameters not found for {args.algo}-{args.env_name}')
        print('the loaded hyperparams ==>', hyperparams)
    else:
        print('the given hyperparams ==>', hyperparams)

    args.n_timesteps = hyperparams['n_timesteps']
    args.patch_dim = hyperparams['patch_dim']
    args.num_blocks = hyperparams['num_blocks']
    args.features_dim = hyperparams['features_dim']
    args.embed_dim_inner = hyperparams['embed_dim_inner']
    args.num_heads_inner = hyperparams['num_heads_inner']
    args.attention_dropout_inner = hyperparams['attention_dropout_inner']
    args.ffn_dropout_inner = hyperparams['ffn_dropout_inner']
    args.embed_dim_outer = hyperparams['embed_dim_outer']
    args.num_heads_outer = hyperparams['num_heads_outer']
    args.attention_dropout_outer = hyperparams['attention_dropout_outer']
    args.ffn_dropout_outer = hyperparams['ffn_dropout_outer']
    activation_fn = {'tanh': nn.Tanh, 'relu': nn.ReLU, 'gelu': nn.GELU}
    args.activation_fn_inner = activation_fn[hyperparams['activation_fn_inner']]
    args.activation_fn_outer = activation_fn[hyperparams['activation_fn_outer']]
    args.activation_fn_other = activation_fn[hyperparams['activation_fn_other']]
    args.dim_expand_inner = hyperparams['dim_expand_inner']
    args.dim_expand_outer = hyperparams['dim_expand_outer']
    args.have_position_encoding = hyperparams['have_position_encoding']
    args.share_tit_blocks = hyperparams['share_tit_blocks']
    print('the updated args ==>', args)

    return args


def load_policy_kwargs(args):
    if args.algo == 'resnet_ppo':
        policy_kwargs = dict(
            features_extractor_class=ResnetFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=512),
            net_arch=[],
        )
    elif args.algo == 'catformer_ppo':
        policy_kwargs = dict(
            features_extractor_class=CatformerFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=512),
            net_arch=[],
        )
    else:  # vanilla_tit and enhanced_tit
        policy_kwargs = dict(
            features_extractor_class=TitFeaturesExtractor,
            features_extractor_kwargs=dict(
                algo=args.algo,
                patch_dim=args.patch_dim,
                num_blocks=args.num_blocks,
                features_dim=args.features_dim,
                embed_dim_inner=args.embed_dim_inner,
                num_heads_inner=args.num_heads_inner,
                attention_dropout_inner=args.attention_dropout_inner,
                ffn_dropout_inner=args.ffn_dropout_inner,
                embed_dim_outer=args.embed_dim_outer,
                num_heads_outer=args.num_heads_outer,
                attention_dropout_outer=args.attention_dropout_outer,
                ffn_dropout_outer=args.ffn_dropout_outer,
                activation_fn_inner=args.activation_fn_inner,
                activation_fn_outer=args.activation_fn_outer,
                activation_fn_other=args.activation_fn_other,
                dim_expand_inner=args.dim_expand_inner,
                dim_expand_outer=args.dim_expand_outer,
                have_position_encoding=args.have_position_encoding,
                share_tit_blocks=args.share_tit_blocks,
            ),
            net_arch=[],
        )
    return policy_kwargs

