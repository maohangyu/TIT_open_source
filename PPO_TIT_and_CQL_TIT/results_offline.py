import os
import time
import numpy as np
import matplotlib.pyplot as plt
import glob
import dataclasses
from d3rlpy_benchmarks.data_loader import load_d4rl_score
from d3rlpy_benchmarks.plot_utils import plot_score_curve
from d3rlpy_benchmarks.utils import get_canonical_algo_name, normalize_d4rl_score


@dataclasses.dataclass(frozen=True)
class ScoreData:
    algo: str
    env: str
    dataset: str
    steps: np.ndarray
    raw_scores: np.ndarray
    scores: np.ndarray


def load_my_score(algo: str, env: str, dataset: str, MY_DIR = "./d3rlpy_logs/"):
    # https://github.com/takuseno/d3rlpy-benchmarks/blob/main/d3rlpy_benchmarks/data_loader.py#L33
    score_list = []
    step_list = []
    for log_dir in glob.glob(os.path.join(MY_DIR, f"CQL_{env}-{dataset}_{algo}_*")):
        if log_dir.endswith('.pt'):
            continue
        with open(os.path.join(log_dir, "environment.csv"), "r") as f:
            data = np.loadtxt(f, delimiter=",", skiprows=1)
            if len(data[:, 2]) < 499:  # discard incomplete data
                continue
            score_list.append(data[:, 2])
            step_list.append(data[:, 1])
    raw_scores = np.array(score_list)
    steps = np.array(step_list)

    if algo == 'cql':
        algo = 'CQL_Reproduced'
    elif algo == 'enhanced_tit_cql':
        algo = 'CQL_TIT_Enhanced'
    elif algo == 'vanilla_tit_cql':
        algo = 'CQL_TIT_Vanilla'

    return ScoreData(
        algo=get_canonical_algo_name(algo),
        env=env,
        dataset=dataset,
        steps=steps,
        raw_scores=raw_scores,
        scores=normalize_d4rl_score(env, raw_scores),
    )


def load_all_d4rl_scores():
    env_name_list = ["halfcheetah", "hopper", "walker2d"]
    type_name_list = ["medium-v0", "medium-replay-v0"][1:]
    algo_name_list = ['CQL', 'cql', 'enhanced_tit_cql']
    for env_name in env_name_list:
        for type_name in type_name_list:
            plt.cla()
            for algo_name in algo_name_list:
                try:
                    print('=='*30, f'load results of {env_name}-{type_name}-{algo_name}')
                    if algo_name == "CQL":  # load baseline CQL
                        score = load_d4rl_score(algo_name, env_name, type_name)  # score.scores.shape ==> (10, 499)
                    else:  # load our implementation of algorithms
                        score = load_my_score(algo_name, env_name, type_name, MY_DIR='./log/')
                    print(score.scores.max(axis=1), np.mean(score.scores.max(axis=1)), np.std(score.scores.max(axis=1)))
                    plot_score_curve(score, window_size=100)
                except:
                    pass
            plt.plot()
            cur_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            plt.savefig('./log/' + env_name + '_' + type_name + '_' + cur_time + '_learning_curve.pdf')


def load_my_score_for_one_experiment(log_dir: str):
    score_list = []
    step_list = []
    with open(os.path.join(log_dir, "environment.csv"), "r") as f:
        data = np.loadtxt(f, delimiter=",", skiprows=1)
        score_list.append(data[:, 2])
        step_list.append(data[:, 1])
    raw_scores = np.array(score_list)
    steps = np.array(step_list)

    algo = log_dir.split('_')[-1]
    env_dataset = log_dir.split('_')[1]
    env = env_dataset.split('-')[0]
    dataset = env_dataset.split('-')[1]
    if algo == 'cql':
        algo = 'CQL_Reproduced'
    elif algo == 'enhanced_tit_cql':
        algo = 'CQL_TIT_Enhanced'
    elif algo == 'vanilla_tit_cql':
        algo = 'CQL_TIT_Vanilla'

    return ScoreData(
        algo=get_canonical_algo_name(algo),
        env=env,
        dataset=dataset,
        steps=steps,
        raw_scores=raw_scores,
        scores=normalize_d4rl_score(env, raw_scores),
    )


def load_all_d4rl_scores_for_one_experiment():
    env_name_list = ["halfcheetah", "hopper", "walker2d"]
    type_name_list = ["medium-v0", "medium-replay-v0"][:1]
    algo_name_list = ['CQL', 'enhanced_tit_cql']
    for env_name in env_name_list:
        for type_name in type_name_list:
            plt.cla()
            for algo_name in algo_name_list:
                    print('=='*30, f'load results of {env_name}-{type_name}-{algo_name}')
                    if algo_name == "CQL":  # load baseline CQL
                        score = load_d4rl_score(algo_name, env_name, type_name)  # score.scores.shape ==> (10, 499)
                        print(score.scores.max(axis=1), np.mean(score.scores.max(axis=1)), np.std(score.scores.max(axis=1)))
                        # pass
                        plot_score_curve(score, window_size=100)
                    else:  # load our implementation of algorithms
                        for log_dir in glob.glob(os.path.join(f"./log/CQL_{env_name}-{type_name}_{algo_name}_*")):
                            if log_dir.endswith('.pt'):
                                continue
                            print('log_dir ==>', log_dir)
                            score = load_my_score_for_one_experiment(log_dir)
                            print(score.scores.max(axis=1), np.mean(score.scores.max(axis=1)), np.std(score.scores.max(axis=1)))
                            # pass
                            plot_score_curve(score, window_size=100)
            plt.plot()
            cur_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            plt.savefig('./log/' + env_name + '_' + type_name + '_' + cur_time + '_learning_curve.pdf')

