import numpy as np


# how to get the score of an expert policy? report the score in the last epoch or the best score during training?
# https://github.com/kzl/decision-transformer/issues/16#issuecomment-890423427
# https://github.com/kzl/decision-transformer/issues/46#issuecomment-1214102788
def normalize_atari_score(env, score):
    Random = {
        'Breakout': 2,
        'Qbert': 164,
        'Pong': -21,
        'Seaquest': 68
    }
    Gamer = {
        'Breakout': 30,
        'Qbert': 13455,
        'Pong': 15,
        'Seaquest': 42055
    }
    min_score = Random[env]
    max_score = Gamer[env]
    normalized_score = 100.0 * (score - min_score) / (max_score - min_score)
    return normalized_score


def read_target_log(file_name, info_type='eval return:'):
    value_list = []
    with open(file_name, 'r') as fp:
        for line in fp.readlines():
            if info_type in line:
                info, value = line.split(info_type)
                value = float(value)
                value_list.append(value)
    return value_list


def read_all_log():
    for env in ['Breakout', 'Pong']:
        for seed in ['123', '231', '312']:
            for algo in ['dt', 'tit']:
                file_name = f'{env}-{algo}_log_{seed}.txt'
                print('='*50, 'file_name ==>', file_name)
                value_list = read_target_log(file_name)
                assert len(value_list) == 5
                print('value_list ==>', value_list)
                return_mean = np.mean(value_list)
                return_std = np.std(value_list)
                print('return_mean/std ==>', return_mean, return_std)
                normalized_return_mean = normalize_atari_score(env, score=return_mean)
                normalized_return_std = normalize_atari_score(env, score=return_std)
                print('normalized_return_mean/std ==>', normalized_return_mean, normalized_return_std)

