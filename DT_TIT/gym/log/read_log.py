import numpy as np
from d3rlpy_benchmarks.utils import normalize_d4rl_score


def read_target_log(file_name, info_type='return_mean'):
    value_list = []
    with open(file_name, 'r') as fp:
        for line in fp.readlines():
            if info_type in line:
                info, value = line.split(':')
                value = float(value)
                value_list.append(value)
    return value_list


def read_all_log():
    for env in ['halfcheetah', 'hopper', 'walker2d']:
        for dataset in ['medium', 'medium-replay']:
            for algo in ['dt', 'tit']:
                file_name = f'{env}-{dataset}-{algo}_log.txt'
                print('='*50, 'file_name ==>', file_name)
                value_list = read_target_log(file_name)
                assert len(value_list) == 60

                return_mean = np.mean([max(value_list[:20]), max(value_list[20:40]), max(value_list[40:])])
                return_std = np.std([max(value_list[:20]), max(value_list[20:40]), max(value_list[40:])])
                print('return_mean/std ==>', return_mean, return_std)

                # https://github.com/takuseno/d3rlpy-benchmarks/blob/main/d3rlpy_benchmarks/utils.py#L16
                normalized_return_mean = normalize_d4rl_score(env, score=return_mean)
                normalized_return_std = normalize_d4rl_score(env, score=return_std)
                print('normalized_return_mean/std ==>', normalized_return_mean, normalized_return_std)


if __name__ == '__main__':
    read_all_log()

