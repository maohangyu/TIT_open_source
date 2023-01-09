import os
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt


def read_evaluation_results(dir_name='./log/'):
    eval_log_path_list = os.listdir(dir_name)
    for log_path in sorted(eval_log_path_list):
        try:
            episode_rewards = np.load(dir_name + log_path + '/eval_episode_rewards.npy')
            print('==' * 20, 'eval_log_path ==>', log_path)
            print('mean/std rewards ==>', np.mean(episode_rewards), np.std(episode_rewards))
        except:
            pass  # if some experiments are still running, skip it


def read_tensorboard_logs(dir_name='./log/', env_name='CartPole-v1'):
    def plot(steps_values, label):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(20, 16))
        ax1 = fig.add_subplot(111)
        ax1.plot([i.step for i in steps_values], [i.value for i in steps_values], label=label)
        ax1.set_xlabel('training step')
        ax1.set_ylabel('mean episode reward')
        plt.legend(loc='lower right')
        plt.show()

    fig = plt.figure(figsize=(20, 16))
    ax1 = fig.add_subplot(111)
    # file_name='./log/CartPole-v1__PPO__0/PPO_1/events.out.tfevents.1665235.8280L-SYS-7049GP-TRT.96687.0'
    for log_path in sorted(os.listdir(dir_name)):
        try:
            if env_name in log_path and 'hypertuning' not in log_path:
                dir_name_temp = dir_name + log_path + '/PPO_1/'
                file_name = dir_name_temp + os.listdir(dir_name_temp)[0]
                ea = event_accumulator.EventAccumulator(file_name)
                ea.Reload()
                print(ea.scalars.Keys())  # ['rollout/ep_len_mean', 'rollout/ep_rew_mean', 'train/approx_kl', ...]
                ep_rew_mean = ea.scalars.Items('rollout/ep_rew_mean')
                print('episode mean rewards ==>', [(i.step, i.value) for i in ep_rew_mean])
                # plot(steps_values=ep_rew_mean, label=log_path)
                ax1.plot([i.step for i in ep_rew_mean], [i.value for i in ep_rew_mean], label=log_path)
        except:
            pass  # if some experiments are still running, skip it
    ax1.set_xlabel('training step')
    ax1.set_ylabel('mean episode reward')
    plt.legend(loc='lower right')
    # plt.show()
    plt.savefig(f'./log/{env_name}_tensorboard.png')


def read_hypertuning_tensorboard_logs(dir_name='./log/', env_name='CartPole-v1__PPO__0__hypertuning'):
    fig = plt.figure(figsize=(20, 16))
    ax1 = fig.add_subplot(111)
    # file_name='./log/CartPole-v1__PPO__0__hypertuning/PPO_1/events.out.tfevents.1665235.8280L-SYS-7049GP-TRT.96687.0'
    for ppo_i in sorted(os.listdir(dir_name + env_name)):
        dir_name_temp = dir_name + env_name + '/' + ppo_i + '/'
        file_name = dir_name_temp + os.listdir(dir_name_temp)[0]
        ea = event_accumulator.EventAccumulator(file_name)
        ea.Reload()
        print(ea.scalars.Keys())  # ['rollout/ep_len_mean', 'rollout/ep_rew_mean', 'train/approx_kl', ...]
        ep_rew_mean = ea.scalars.Items('rollout/ep_rew_mean')
        print('episode mean rewards ==>', [(i.step, i.value) for i in ep_rew_mean])
        ax1.plot([i.step for i in ep_rew_mean], [i.value for i in ep_rew_mean], label=dir_name_temp)
    ax1.set_xlabel('training step')
    ax1.set_ylabel('mean episode reward')
    plt.legend()
    # plt.show()
    plt.savefig(f'./log/{env_name}_tensorboard.png')

