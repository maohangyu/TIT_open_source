import numpy as np
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from utils import linear_schedule, update_args, load_policy_kwargs
from network import OFENetActorCriticPolicy, D2RLNetActorCriticPolicy


def make_image_agent(env_name, algo, seed, args):
    log_folder = args.log_folder
    device = args.device

    policy_type = 'CnnPolicy'
    if algo == 'ppo':  # use default setting
        # https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml
        # https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#id2 [Atari Games]
        env = VecFrameStack(make_atari_env(env_name, n_envs=8, seed=seed), n_stack=4)
        agent = PPO(policy=policy_type, env=env, tensorboard_log=log_folder, verbose=1, seed=seed,
                    n_steps=128, n_epochs=4, batch_size=256, learning_rate=linear_schedule(2.5e-4),
                    clip_range=linear_schedule(0.1), vf_coef=0.5, ent_coef=0.01, device=device)
    elif algo in ['resnet_ppo', 'catformer_ppo']:
        policy_kwargs = load_policy_kwargs(args)
        env = VecFrameStack(make_atari_env(env_name, n_envs=8, seed=seed), n_stack=4)
        agent = PPO(policy=policy_type, env=env, tensorboard_log=log_folder, verbose=1, seed=seed,
                    n_steps=128, n_epochs=4, batch_size=256, learning_rate=linear_schedule(2.5e-4),
                    clip_range=linear_schedule(0.1), vf_coef=0.5, ent_coef=0.01, device=device,
                    policy_kwargs=policy_kwargs)
    else:  # 'vanilla_tit_ppo', 'enhanced_tit_ppo'
        policy_kwargs = load_policy_kwargs(args)
        env = VecFrameStack(make_atari_env(env_name, n_envs=8, seed=seed), n_stack=4)  # keep the same as SB3-PPO
        agent = PPO(policy=policy_type, env=env, tensorboard_log=log_folder, verbose=1, seed=seed,
                    n_steps=128, n_epochs=4, batch_size=64, learning_rate=linear_schedule(2.5e-4),  # remove
                    clip_range=linear_schedule(0.1), vf_coef=0.5, ent_coef=0.01, device=device,
                    policy_kwargs=policy_kwargs)
        # we don't use linear_schedule for learning_rate/clip_range,
        # we also use batch_size=64 rather than 256 to avoid CUDA OOM,
        # we will show that even with these un-tuned hyperparameters, TIT works well!

    return agent


def make_array_agent(env_name, algo, seed, args):
    log_folder = args.log_folder
    device = args.device

    if algo == 'ppo':  # use default setting
        policy_type = 'MlpPolicy'
        agent = PPO(policy=policy_type, env=env_name, tensorboard_log=log_folder, verbose=1, seed=seed,
                    device=device)
    elif algo in ['ofe_ppo', 'd2rl_ppo']:
        policy_type = OFENetActorCriticPolicy if algo=='ofe_ppo' else D2RLNetActorCriticPolicy
        agent = PPO(policy=policy_type, env=env_name, tensorboard_log=log_folder, verbose=1, seed=seed,
                    device=device)
    else:  # 'vanilla_tit_ppo', 'enhanced_tit_ppo'
        policy_type = 'MlpPolicy'
        policy_kwargs = load_policy_kwargs(args)
        agent = PPO(policy=policy_type, env=env_name, tensorboard_log=log_folder, verbose=1, seed=seed,
                    device=device, policy_kwargs=policy_kwargs)

    return agent


def train(env_name, algo, n_timesteps, seed, args):
    args.log_folder = './log/' + env_name + '__' + algo + '__' + str(seed) + '__running'

    if 'NoFrameskip' in env_name:
        agent = make_image_agent(env_name, algo, seed, args)
        print('make_image_agent')
    else:
        agent = make_array_agent(env_name, algo, seed, args)
    print('==' * 20, 'policy structure ==>', agent.policy)
    print('==' * 20, 'number of parameters: %d' % sum(p.numel() for p in agent.policy.parameters()))
    print('==' * 20, 'observation_space.shape ==>', agent.get_env().observation_space.shape)  # (4,) or (4, 84, 84)

    # agent.learn(total_timesteps=train_step, eval_freq=int(train_step // 4),
    #             eval_env=gym.make(env_name), n_eval_episodes=100, eval_log_path=log_path)
    agent.learn(total_timesteps=n_timesteps)
    agent.save(args.log_folder + '/final_model')

    episode_rewards, episode_lengths = evaluate_policy(agent.policy, agent.get_env(),
                                                       n_eval_episodes=100, return_episode_rewards=True)
    print('==' * 20, 'mean/std episode_rewards ==>', np.mean(episode_rewards), np.std(episode_rewards))
    np.save(args.log_folder + '/eval_episode_rewards.npy', episode_rewards)
    np.save(args.log_folder + '/eval_episode_lengths.npy', episode_lengths)


if __name__ == '__main__':
    # import torch
    # torch.set_num_threads(3)
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", help="Environment ID", type=str, default="CartPole-v1")
    parser.add_argument("--algo", help="RL Algorithm", type=str, default="enhanced_tit_ppo",
        choices=['ppo', 'ofe_ppo', 'd2rl_ppo', 'resnet_ppo', 'catformer_ppo', 'vanilla_tit_ppo', 'enhanced_tit_ppo'])
    parser.add_argument("--device", help="PyTorch device (ex: cpu, cuda:0, cuda:1, ...)", type=str, default="auto")
    parser.add_argument("--log-folder", help="Log folder", type=str, default="./log/")
    #
    parser.add_argument("--n-timesteps", help="Timesteps to run the env for one trial", type=int, default=100000)
    parser.add_argument("--patch-dim", help="patch_dim", type=int, default=6)
    parser.add_argument("--num-blocks", help="how many Transformer blocks to use", type=int, default=2)
    parser.add_argument("--features-dim", help="features_dim of last layer", type=int, default=64)
    parser.add_argument("--embed-dim-inner", help="embed_dim_inner", type=int, default=8)
    parser.add_argument("--num-heads-inner", help="num_heads_inner", type=int, default=4)
    parser.add_argument("--attention-dropout-inner", help="attention_dropout_inner", type=float, default=0.0)
    parser.add_argument("--ffn-dropout-inner", help="ffn_dropout_inner", type=float, default=0.0)
    parser.add_argument("--embed-dim-outer", help="embed_dim_outer", type=int, default=64)
    parser.add_argument("--num-heads-outer", help="num_heads_outer", type=int, default=4)
    parser.add_argument("--attention-dropout-outer", help="attention_dropout_outer", type=float, default=0.0)
    parser.add_argument("--ffn-dropout-outer", help="ffn_dropout_outer", type=float, default=0.0)
    parser.add_argument("--activation-fn-inner", help="activation_function_inner", default=None)
    parser.add_argument("--activation-fn-outer", help="activation_function_outer", default=None)
    parser.add_argument("--activation-fn-other", help="activation_function_other", default=None)
    args = parser.parse_args()

    if args.algo == 'ppo':
        env_name_list = [
            'Acrobot-v1', 'CartPole-v1', 'MountainCar-v0',
            'Ant-v3', 'Hopper-v3', 'Walker2d-v3',
            'BreakoutNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 'PongNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4',
        ]
        train_step_list = [
            100000, 100000, 100000,
            1000000, 1000000, 1000000,
            10000000, 10000000, 10000000, 10000000,
        ]
        for env_name, n_timesteps in zip(env_name_list, train_step_list):
            for seed in range(5):
                set_random_seed(seed)
                train(env_name, args.algo, n_timesteps, seed, args)
    elif args.algo in ['ofe_ppo', 'd2rl_ppo']:
        env_name_list = [
            'Acrobot-v1', 'CartPole-v1', 'MountainCar-v0',
            'Ant-v3', 'Hopper-v3', 'Walker2d-v3',
        ]
        train_step_list = [
            100000, 100000, 100000,
            1000000, 1000000, 1000000,
        ]
        for env_name, n_timesteps in zip(env_name_list, train_step_list):
            for seed in range(5):
                set_random_seed(seed)
                train(env_name, args.algo, n_timesteps, seed, args)
    elif args.algo in ['resnet_ppo', 'catformer_ppo']:
        env_name_list = [
            'BreakoutNoFrameskip-v4', 'MsPacmanNoFrameskip-v4', 'PongNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4',
        ]
        train_step_list = [
            10000000, 10000000, 10000000, 10000000,
        ]
        for env_name, n_timesteps in zip(env_name_list, train_step_list):
            for seed in range(5):
                set_random_seed(seed)
                train(env_name, args.algo, n_timesteps, seed, args)
    elif args.algo in ['vanilla_tit_ppo', 'enhanced_tit_ppo']:
        args = update_args(args)
        experiment_count = 2 if 'NoFrameskip' in args.env_name else 5
        for seed in range(experiment_count):
            set_random_seed(seed)
            train(args.env_name, args.algo, args.n_timesteps, seed, args)

