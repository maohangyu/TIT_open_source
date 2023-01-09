'''
based on https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/hyperparams_opt.py
'''

import optuna
import numpy as np
from torch import nn as nn
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from utils import linear_schedule
from network import TitFeaturesExtractor


def sample_dqn_params(trial: optuna.Trial):
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 100, 128, 256, 512])
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(5e4), int(1e5), int(1e6)])
    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 128, 256, 1000])
    subsample_steps = trial.suggest_categorical("subsample_steps", [1, 2, 4, 8])
    gradient_steps = max(train_freq // subsample_steps, 1)
    exploration_final_eps = trial.suggest_uniform("exploration_final_eps", 0, 0.2)
    exploration_fraction = trial.suggest_uniform("exploration_fraction", 0, 0.5)
    target_update_interval = trial.suggest_categorical("target_update_interval", [1, 1000, 5000, 10000, 15000, 20000])
    learning_starts = trial.suggest_categorical("learning_starts", [0, 1000, 5000, 10000, 20000])
    net_arch = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])
    net_arch = {"tiny": [64], "small": [64, 64], "medium": [256, 256]}[net_arch]

    hyperparams = {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "exploration_final_eps": exploration_final_eps,
        "exploration_fraction": exploration_fraction,
        "target_update_interval": target_update_interval,
        "learning_starts": learning_starts,
        "policy_kwargs": dict(net_arch=net_arch),
    }

    if trial.using_her_replay_buffer:
        hyperparams = sample_her_params(trial, hyperparams)

    return hyperparams


def sample_ddpg_params(trial: optuna.Trial):
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 100, 128, 256, 512, 1024, 2048])
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 32, 64, 128, 256, 512])
    gradient_steps = train_freq
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05, 0.08])  # Polyak coeff
    # NOTE: Add "verybig" to net_arch when tuning HER (see TD3)
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big"])
    net_arch = {"small": [64, 64], "medium": [256, 256], "big": [400, 300]}[net_arch]
    # activation_fn = trial.suggest_categorical('activation_fn', [nn.Tanh, nn.ReLU, nn.ELU, nn.LeakyReLU])
    noise_type = trial.suggest_categorical("noise_type", ["ornstein-uhlenbeck", "normal", None])
    noise_std = trial.suggest_uniform("noise_std", 0, 1)

    hyperparams = {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "tau": tau,
        "policy_kwargs": dict(net_arch=net_arch),
    }

    if noise_type == "normal":
        hyperparams["action_noise"] = NormalActionNoise(
            mean=np.zeros(trial.n_actions), sigma=noise_std * np.ones(trial.n_actions)
        )
    elif noise_type == "ornstein-uhlenbeck":
        hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(trial.n_actions), sigma=noise_std * np.ones(trial.n_actions)
        )

    if trial.using_her_replay_buffer:
        hyperparams = sample_her_params(trial, hyperparams)

    return hyperparams


def sample_td3_params(trial: optuna.Trial):
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 100, 128, 256, 512, 1024, 2048])
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 32, 64, 128, 256, 512])
    gradient_steps = train_freq
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05, 0.08])  # Polyak coeff
    # NOTE: Add "verybig" to net_arch when tuning HER
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big"])
    net_arch = {"small": [64, 64], "medium": [256, 256], "big": [400, 300],
                # "verybig": [256, 256, 256],  # Uncomment for tuning HER
                }[net_arch]
    # activation_fn = trial.suggest_categorical('activation_fn', [nn.Tanh, nn.ReLU, nn.ELU, nn.LeakyReLU])
    noise_type = trial.suggest_categorical("noise_type", ["ornstein-uhlenbeck", "normal", None])
    noise_std = trial.suggest_uniform("noise_std", 0, 1)

    hyperparams = {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "tau": tau,
        "policy_kwargs": dict(net_arch=net_arch),
    }

    if noise_type == "normal":
        hyperparams["action_noise"] = NormalActionNoise(
            mean=np.zeros(trial.n_actions), sigma=noise_std * np.ones(trial.n_actions)
        )
    elif noise_type == "ornstein-uhlenbeck":
        hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(trial.n_actions), sigma=noise_std * np.ones(trial.n_actions)
        )

    if trial.using_her_replay_buffer:
        hyperparams = sample_her_params(trial, hyperparams)

    return hyperparams


def sample_her_params(trial: optuna.Trial, hyperparams):
    her_kwargs = trial.her_kwargs.copy()
    her_kwargs["n_sampled_goal"] = trial.suggest_int("n_sampled_goal", 1, 5)
    her_kwargs["goal_selection_strategy"] = trial.suggest_categorical("goal_selection_strategy",
                                                                      ["final", "episode", "future"])
    her_kwargs["online_sampling"] = trial.suggest_categorical("online_sampling", [True, False])
    hyperparams["replay_buffer_kwargs"] = her_kwargs
    return hyperparams


def sample_ppo_params(trial: optuna.Trial):
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    lr_schedule = "constant"
    # Uncomment to enable learning rate schedule
    # lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])

    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    ent_coef = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)
    vf_coef = trial.suggest_uniform("vf_coef", 0, 1)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])

    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    # Independent networks usually work best when not working with images
    net_arch = {"small": [dict(pi=[64, 64], vf=[64, 64])], "medium": [dict(pi=[256, 256], vf=[256, 256])]}[net_arch]
    # activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]
    # Orthogonal initialization
    ortho_init = False
    # ortho_init = trial.suggest_categorical('ortho_init', [False, True])
    # Uncomment for gSDE (continuous actions)
    # log_std_init = trial.suggest_uniform("log_std_init", -4, 1)
    # Uncomment for gSDE (continuous action)
    # sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])

    # TODO: account when using multiple envs
    if batch_size > n_steps:
        batch_size = n_steps

    return {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "n_steps": n_steps,
        "gae_lambda": gae_lambda,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "clip_range": clip_range,
        "max_grad_norm": max_grad_norm,
        "policy_kwargs": dict(
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            # log_std_init=log_std_init,
        ),
        # "sde_sample_freq": sde_sample_freq,
    }


def sample_tit_ppo_params_old(trial: optuna.Trial, env_info='Atari-BreakoutNoFrameskip-v4', algo='vanilla_tit_ppo'):
    env_type, env_name, version = env_info.split('-')
    if env_type == 'Atari':
        patch_dim = trial.suggest_categorical("patch_dim", [6, 12, 42])
        num_blocks = 2 if True else trial.suggest_categorical("num_blocks", [2, 3, 4])
    elif env_type == 'ClassicControl' or env_type == 'MuJoCo':
        patch_dim = 1
        num_blocks = 2
    else:
        raise ValueError('env_type should be either Atari, ClassicControl or Mujoco.')

    if env_type == 'ClassicControl' or env_type == 'Atari':
        features_dim = trial.suggest_categorical("features_dim", [32, 64, 128, 256, 512])
        embed_dim_inner = trial.suggest_categorical("embed_dim_inner", [16, 32, 64, 128, 256])
        num_heads_inner = trial.suggest_categorical("num_heads_inner", [2, 4, 8])
        assert embed_dim_inner % num_heads_inner == 0
        embed_dim_outer = embed_dim_inner
        num_heads_outer = trial.suggest_categorical("num_heads_outer", [2, 4, 8])
        assert embed_dim_outer % num_heads_outer == 0
        attention_dropout_inner = 0.0 if True else trial.suggest_categorical("attention_dropout_inner", [0.0, 0.1])
        ffn_dropout_inner = 0.0 if True else trial.suggest_categorical("ffn_dropout_inner", [0.0, 0.1])
        attention_dropout_outer = 0.0 if True else trial.suggest_categorical("attention_dropout_outer", [0.0, 0.1])
        ffn_dropout_outer = 0.0 if True else trial.suggest_categorical("ffn_dropout_outer", [0.0, 0.1])
    elif env_type == 'MuJoCo':
        features_dim = trial.suggest_categorical("features_dim", [32, 64, 128, 256, 512])

        embed_dim_inner = trial.suggest_categorical("embed_dim_inner", [1, 2, 4, 8, 16, 32])
        '''
        bad implementation #1
        if embed_dim_inner == 1:
            num_heads_inner = 1
        if embed_dim_inner == 2:
            num_heads_inner = trial.suggest_categorical("num_heads_inner", [1, 2])
        if embed_dim_inner == 4:
            num_heads_inner = trial.suggest_categorical("num_heads_inner", [1, 2, 4])
        if embed_dim_inner == 8:
            num_heads_inner = trial.suggest_categorical("num_heads_inner", [1, 2, 4, 8])
        if embed_dim_inner == 16:
            num_heads_inner = trial.suggest_categorical("num_heads_inner", [1, 2, 4, 8, 16])
        if embed_dim_inner == 32:
            num_heads_inner = trial.suggest_categorical("num_heads_inner", [1, 2, 4, 8, 16, 32])
        assert embed_dim_inner % num_heads_inner == 0
        ==> ValueError: CategoricalDistribution does not support dynamic value space.
        https://github.com/optuna/optuna/issues/372
        
        bad implementation #2
        num_heads_inner = 100
        while embed_dim_inner % num_heads_inner != 0:
            num_heads_inner = trial.suggest_categorical("num_heads_inner", [1, 2, 4, 8, 16, 32])
            # print('num_heads_inner ==>', num_heads_inner)  ===> this will always print the same value!
        '''
        if embed_dim_inner == 1:
            num_heads_inner = 1
        elif embed_dim_inner == 2:
            num_heads_inner = 2
        else:
            num_heads_inner = trial.suggest_categorical("num_heads_inner", [1, 2, 4])
        assert embed_dim_inner % num_heads_inner == 0

        embed_dim_outer = embed_dim_inner
        if embed_dim_outer == 1:
            num_heads_outer = 1
        elif embed_dim_outer == 2:
            num_heads_outer = 2
        else:
            num_heads_outer = trial.suggest_categorical("num_heads_outer", [1, 2, 4])
        assert embed_dim_outer % num_heads_outer == 0

        attention_dropout_inner = trial.suggest_categorical("attention_dropout_inner", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        ffn_dropout_inner = trial.suggest_categorical("ffn_dropout_inner", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        attention_dropout_outer = trial.suggest_categorical("attention_dropout_outer", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        ffn_dropout_outer = trial.suggest_categorical("ffn_dropout_outer", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    activation_fn = {'gelu': nn.GELU, "tanh": nn.Tanh, 'relu': nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}
    activation_fn_inner = 'gelu' if True else trial.suggest_categorical("activation_fn_inner", ['tanh', 'gelu'])
    activation_fn_outer = 'gelu' if True else trial.suggest_categorical("activation_fn_outer", ['tanh', 'gelu'])
    activation_fn_other = 'tanh' if False else trial.suggest_categorical("activation_fn_other", ['tanh', 'relu'])
    activation_fn_inner = activation_fn[activation_fn_inner]
    activation_fn_outer = activation_fn[activation_fn_outer]
    activation_fn_other = activation_fn[activation_fn_other]
    # default of A2C, PPO: nn.Tanh; default of DQN/DDPG/SAC/TD3: nn.ReLU
    # PPO uses nn.Tanh as default, which performs much better than nn.ReLU for [continuous action] envs like Mujoco.

    return {
        "policy_kwargs": dict(
            features_extractor_class=TitFeaturesExtractor,
            features_extractor_kwargs=dict(
                algo=algo,
                patch_dim=patch_dim,
                num_blocks=num_blocks,
                features_dim=features_dim,
                embed_dim_inner=embed_dim_inner,
                num_heads_inner=num_heads_inner,
                attention_dropout_inner=attention_dropout_inner,
                ffn_dropout_inner=ffn_dropout_inner,
                embed_dim_outer=embed_dim_outer,
                num_heads_outer=num_heads_outer,
                attention_dropout_outer=attention_dropout_outer,
                ffn_dropout_outer=ffn_dropout_outer,
                activation_fn_inner=activation_fn_inner,
                activation_fn_outer=activation_fn_outer,
                activation_fn_other=activation_fn_other,
            ),
            net_arch=[],
        )
    }


def sample_tit_ppo_params(trial: optuna.Trial, env_info='Atari-BreakoutNoFrameskip-v4', algo='vanilla_tit_ppo'):
    env_type, env_name, version = env_info.split('-')
    if env_type == 'Atari':
        patch_dim = trial.suggest_categorical("patch_dim", [6, 12, 42])
        num_blocks = 2 if True else trial.suggest_categorical("num_blocks", [2, 3, 4])
    elif env_type == 'ClassicControl':
        patch_dim = 1
        num_blocks = 2
    elif env_type == 'MuJoCo':  # 'Ant-v3', 'Hopper-v3', 'Walker2d-v3'
        if env_name == 'Ant':
            patch_dim = 111 if True else trial.suggest_categorical("patch_dim", [1, 11, 111])
        elif env_name == 'Hopper':
            patch_dim = 11 if True else trial.suggest_categorical("patch_dim", [1, 11])
        elif env_name == 'Walker2d':
            patch_dim = 17 if True else trial.suggest_categorical("patch_dim", [1, 11, 17])
        else:
            raise ValueError('env_name is not in [Ant-v3, Hopper-v3, Walker2d-v3].')
        num_blocks = 1 if False else trial.suggest_categorical("num_blocks", [1, 2])
    else:
        raise ValueError('env_type should be either Atari, ClassicControl or Mujoco.')

    features_dim = trial.suggest_categorical("features_dim", [32, 64, 128, 256])
    embed_dim_inner = trial.suggest_categorical("embed_dim_inner", [32, 64, 128, 256])
    num_heads_inner = 1 if True else trial.suggest_categorical("num_heads_inner", [1, 2, 4])
    assert embed_dim_inner % num_heads_inner == 0

    embed_dim_outer = embed_dim_inner
    num_heads_outer = 1 if False else trial.suggest_categorical("num_heads_outer", [1, 2, 4])
    assert embed_dim_outer % num_heads_outer == 0

    attention_dropout_inner = 0.0 if True else trial.suggest_categorical("attention_dropout_inner", [0.0, 0.1])
    ffn_dropout_inner = 0.0 if True else trial.suggest_categorical("ffn_dropout_inner", [0.0, 0.1])
    attention_dropout_outer = 0.0 if True else trial.suggest_categorical("attention_dropout_outer", [0.0, 0.1])
    ffn_dropout_outer = 0.0 if True else trial.suggest_categorical("ffn_dropout_outer", [0.0, 0.1])

    activation_fn = {'gelu': nn.GELU, "tanh": nn.Tanh, 'relu': nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}
    activation_fn_inner = 'gelu' if True else trial.suggest_categorical("activation_fn_inner", ['tanh', 'gelu'])
    activation_fn_outer = 'gelu' if True else trial.suggest_categorical("activation_fn_outer", ['tanh', 'gelu'])
    activation_fn_other = 'tanh' if False else trial.suggest_categorical("activation_fn_other", ['tanh', 'relu', 'gelu'])
    activation_fn_inner = activation_fn[activation_fn_inner]
    activation_fn_outer = activation_fn[activation_fn_outer]
    activation_fn_other = activation_fn[activation_fn_other]
    # default of A2C, PPO: nn.Tanh; default of DQN/DDPG/SAC/TD3: nn.ReLU
    # PPO uses nn.Tanh as default, which performs much better than nn.ReLU for [continuous action] envs like Mujoco.

    dim_expand_inner = 1 if True else trial.suggest_categorical("dim_expand_inner", [1, 2, 4])
    dim_expand_outer = 1 if False else trial.suggest_categorical("dim_expand_outer", [1, 2, 4])
    have_position_encoding = 0 if True else trial.suggest_categorical("have_position_encoding", [0, 1])
    share_tit_blocks = 0 if True else trial.suggest_categorical("share_tit_blocks", [0, 1])

    return {
        "policy_kwargs": dict(
            features_extractor_class=TitFeaturesExtractor,
            features_extractor_kwargs=dict(
                algo=algo,
                patch_dim=patch_dim,
                num_blocks=num_blocks,
                features_dim=features_dim,
                embed_dim_inner=embed_dim_inner,
                num_heads_inner=num_heads_inner,
                attention_dropout_inner=attention_dropout_inner,
                ffn_dropout_inner=ffn_dropout_inner,
                embed_dim_outer=embed_dim_outer,
                num_heads_outer=num_heads_outer,
                attention_dropout_outer=attention_dropout_outer,
                ffn_dropout_outer=ffn_dropout_outer,
                activation_fn_inner=activation_fn_inner,
                activation_fn_outer=activation_fn_outer,
                activation_fn_other=activation_fn_other,
                dim_expand_inner=dim_expand_inner,
                dim_expand_outer=dim_expand_outer,
                have_position_encoding=have_position_encoding,
                share_tit_blocks=share_tit_blocks,
            ),
            net_arch=[],
        )
    }


def sample_tit_cql_params(trial: optuna.Trial, env_name='halfcheetah-medium-v0'):
    env_name = env_name.split('-')[0]
    if env_name == 'halfcheetah':
        patch_dim = 17 if True else trial.suggest_categorical("patch_dim", [1, 17])
    elif env_name == 'hopper':
        patch_dim = 11 if True else trial.suggest_categorical("patch_dim", [1, 11])
    elif env_name == 'walker2d':
        patch_dim = 17 if True else trial.suggest_categorical("patch_dim", [1, 17])
    else:
        raise ValueError('env_name should be either halfcheetah, hopper or walker2d.')
    num_blocks = 1 if False else trial.suggest_categorical("num_blocks", [1, 2])

    features_dim = trial.suggest_categorical("features_dim", [128, 256, 512, 1024])
    embed_dim_inner = trial.suggest_categorical("embed_dim_inner", [32, 64, 128, 256])
    num_heads_inner = 1 if True else trial.suggest_categorical("num_heads_inner", [1, 2, 4])
    assert embed_dim_inner % num_heads_inner == 0

    embed_dim_outer = embed_dim_inner
    num_heads_outer = 1 if False else trial.suggest_categorical("num_heads_outer", [1, 2, 4])
    assert embed_dim_outer % num_heads_outer == 0

    attention_dropout_inner = 0.0 if True else trial.suggest_categorical("attention_dropout_inner", [0.0, 0.1])
    ffn_dropout_inner = 0.0 if True else trial.suggest_categorical("ffn_dropout_inner", [0.0, 0.1])
    attention_dropout_outer = 0.0 if False else trial.suggest_categorical("attention_dropout_outer", [0.0, 0.1])
    ffn_dropout_outer = 0.0 if False else trial.suggest_categorical("ffn_dropout_outer", [0.0, 0.1])

    activation_fn_inner = 'gelu' if True else trial.suggest_categorical("activation_fn_inner", ['tanh', 'gelu'])
    activation_fn_outer = 'gelu' if True else trial.suggest_categorical("activation_fn_outer", ['tanh', 'gelu'])
    activation_fn_other = 'tanh' if False else trial.suggest_categorical("activation_fn_other", ['tanh', 'relu', 'gelu'])

    dim_expand_inner = 1 if True else trial.suggest_categorical("dim_expand_inner", [1, 2, 4])
    dim_expand_outer = 1 if True else trial.suggest_categorical("dim_expand_outer", [1, 2, 4])
    have_position_encoding = 0 if True else trial.suggest_categorical("have_position_encoding", [0, 1])
    share_tit_blocks = 0 if False else trial.suggest_categorical("share_tit_blocks", [0, 1])

    return dict(
        n_timesteps=500000,
        patch_dim=patch_dim,
        num_blocks=num_blocks,
        features_dim=features_dim,
        embed_dim_inner=embed_dim_inner,
        num_heads_inner=num_heads_inner,
        attention_dropout_inner=attention_dropout_inner,
        ffn_dropout_inner=ffn_dropout_inner,
        embed_dim_outer=embed_dim_outer,
        num_heads_outer=num_heads_outer,
        attention_dropout_outer=attention_dropout_outer,
        ffn_dropout_outer=ffn_dropout_outer,
        activation_fn_inner=activation_fn_inner,
        activation_fn_outer=activation_fn_outer,
        activation_fn_other=activation_fn_other,
        dim_expand_inner=dim_expand_inner,
        dim_expand_outer=dim_expand_outer,
        have_position_encoding=have_position_encoding,
        share_tit_blocks=share_tit_blocks,
    )


def sample_tit_cql_params_v2(trial: optuna.Trial, env_name='halfcheetah-medium-v0'):
    env_name = env_name.split('-')[0]
    if env_name == 'halfcheetah':
        patch_dim = 17 if True else trial.suggest_categorical("patch_dim", [1, 17])
    elif env_name == 'hopper':
        patch_dim = 11 if True else trial.suggest_categorical("patch_dim", [1, 11])
    elif env_name == 'walker2d':
        patch_dim = 17 if True else trial.suggest_categorical("patch_dim", [1, 17])
    else:
        raise ValueError('env_name should be either halfcheetah, hopper or walker2d.')
    num_blocks = 1 if False else trial.suggest_categorical("num_blocks", [1, 2])

    features_dim = trial.suggest_categorical("features_dim", [128, 256, 512, 1024])
    embed_dim_inner = trial.suggest_categorical("embed_dim_inner", [32, 64, 128, 256])
    num_heads_inner = 1 if True else trial.suggest_categorical("num_heads_inner", [1, 2, 4])
    assert embed_dim_inner % num_heads_inner == 0

    embed_dim_outer = embed_dim_inner
    num_heads_outer = 1 if False else trial.suggest_categorical("num_heads_outer", [1, 2, 4])
    assert embed_dim_outer % num_heads_outer == 0

    attention_dropout_inner = 0.0 if True else trial.suggest_categorical("attention_dropout_inner", [0.0, 0.1])
    ffn_dropout_inner = 0.0 if True else trial.suggest_categorical("ffn_dropout_inner", [0.0, 0.1])
    attention_dropout_outer = 0.0 if True else trial.suggest_categorical("attention_dropout_outer", [0.0, 0.1])
    ffn_dropout_outer = 0.0 if True else trial.suggest_categorical("ffn_dropout_outer", [0.0, 0.1])

    activation_fn_inner = 'gelu' if True else trial.suggest_categorical("activation_fn_inner", ['tanh', 'gelu'])
    activation_fn_outer = 'gelu' if True else trial.suggest_categorical("activation_fn_outer", ['tanh', 'gelu'])
    activation_fn_other = 'tanh' if False else trial.suggest_categorical("activation_fn_other", ['tanh', 'relu', 'gelu'])

    dim_expand_inner = 1 if True else trial.suggest_categorical("dim_expand_inner", [1, 2, 4])
    dim_expand_outer = 1 if False else trial.suggest_categorical("dim_expand_outer", [1, 2, 4])
    have_position_encoding = 0 if False else trial.suggest_categorical("have_position_encoding", [0, 1])
    share_tit_blocks = 0 if False else trial.suggest_categorical("share_tit_blocks", [0, 1])

    return dict(
        n_timesteps=500000,
        patch_dim=patch_dim,
        num_blocks=num_blocks,
        features_dim=features_dim,
        embed_dim_inner=embed_dim_inner,
        num_heads_inner=num_heads_inner,
        attention_dropout_inner=attention_dropout_inner,
        ffn_dropout_inner=ffn_dropout_inner,
        embed_dim_outer=embed_dim_outer,
        num_heads_outer=num_heads_outer,
        attention_dropout_outer=attention_dropout_outer,
        ffn_dropout_outer=ffn_dropout_outer,
        activation_fn_inner=activation_fn_inner,
        activation_fn_outer=activation_fn_outer,
        activation_fn_other=activation_fn_other,
        dim_expand_inner=dim_expand_inner,
        dim_expand_outer=dim_expand_outer,
        have_position_encoding=have_position_encoding,
        share_tit_blocks=share_tit_blocks,
    )


HYPERPARAMS_SAMPLER = {
    "dqn": sample_dqn_params,
    "ddpg": sample_ddpg_params,
    "td3": sample_td3_params,
    "ppo": sample_ppo_params,
    'tit_ppo': sample_tit_ppo_params,
    'enhanced_tit_cql': sample_tit_cql_params_v2,
}
