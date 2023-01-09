import argparse
import d3rlpy
from sklearn.model_selection import train_test_split
from offline_network import MyCustomEncoderFactory
from utils import update_args


def make_agent(args):
    if "medium-v0" in args.env_name:
        conservative_weight = 10.0
    else:
        conservative_weight = 5.0

    # https://d3rlpy.readthedocs.io/en/v1.1.1/references/network_architectures.html
    if args.algo in ['vanilla_tit_cql', 'enhanced_tit_cql']:
        actor_encoder = MyCustomEncoderFactory(
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
            share_tit_blocks=args.share_tit_blocks
        )
        encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256])
        agent = d3rlpy.algos.CQL(actor_learning_rate=1e-4,
                                 critic_learning_rate=3e-4,
                                 temp_learning_rate=1e-4,
                                 actor_encoder_factory=actor_encoder,
                                 critic_encoder_factory=encoder,
                                 batch_size=256,
                                 n_action_samples=10,
                                 alpha_learning_rate=0.0,
                                 conservative_weight=conservative_weight,
                                 use_gpu=d3rlpy.gpu.Device(idx=int(args.device)))
    elif args.algo == 'cql':
        encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256])
        agent = d3rlpy.algos.CQL(actor_learning_rate=1e-4,
                                 critic_learning_rate=3e-4,
                                 temp_learning_rate=1e-4,
                                 actor_encoder_factory=encoder,
                                 critic_encoder_factory=encoder,
                                 batch_size=256,
                                 n_action_samples=10,
                                 alpha_learning_rate=0.0,
                                 conservative_weight=conservative_weight,
                                 use_gpu=args.device)

    print('agent ==>', agent)
    return agent


def train(env_name, seed, args):
    # https://github.com/takuseno/d3rlpy/blob/master/reproductions/offline/cql.py
    dataset, env = d3rlpy.datasets.get_dataset(env_name)
    print("len(dataset):", len(dataset), type(dataset))
    print("len(dataset[0]):", len(dataset), type(dataset[0]))

    # fix seed
    d3rlpy.seed(seed)
    env.seed(seed)

    agent = make_agent(args)
    agent.build_with_dataset(dataset)
    print('agent.impl._policy ==>', agent.impl._policy)
    print('agent.impl._q_func ==>', agent.impl._q_func)

    _, test_episodes = train_test_split(dataset, test_size=0.2)
    results = agent.fit(dataset,
              eval_episodes=test_episodes,
              n_steps=args.n_timesteps,
              n_steps_per_epoch=1000,
              save_interval=10,
              scorers={
                  'environment': d3rlpy.metrics.evaluate_on_environment(env),
                  'value_scale': d3rlpy.metrics.average_value_estimation_scorer,
              },
              experiment_name=f"CQL_{env_name}_{args.algo}_{seed}",
              logdir=args.log_folder,
              show_progress=args.show_progress)

    agent.save_policy(f"{args.log_folder}/CQL_{env_name}_{args.algo}_{seed}.pt")  # save greedy-policy as TorchScript

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", help="Environment ID", type=str, default="halfcheetah-medium-v0",
                        choices=['halfcheetah-medium-v0', 'hopper-medium-v0', 'walker2d-medium-v0',
                                 'halfcheetah-medium-replay-v0', 'hopper-medium-replay-v0', 'walker2d-medium-replay-v0'])
    parser.add_argument("--algo", help="RL Algorithm", type=str, default="cql", choices=['cql', 'enhanced_tit_cql'])
    parser.add_argument("--device", help="PyTorch device (ex: cpu, cuda:0, cuda:1, ...)", default=True)
    parser.add_argument("--log-folder", help="Log folder", type=str, default="./log/")
    parser.add_argument("--show-progress", help="flag to show progress bar for iterations", default=True)
    #
    parser.add_argument("--n-timesteps", help="Timesteps to run the env for one trial", type=int, default=500000)
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

    if args.algo == 'cql':
        env_name_list = [
            'halfcheetah-medium-v0', 'hopper-medium-v0', 'walker2d-medium-v0',
            'halfcheetah-medium-replay-v0', 'hopper-medium-replay-v0', 'walker2d-medium-replay-v0'
        ]
        for env_name in env_name_list:
            for seed in range(5):
                train(env_name, seed, args)
    elif args.algo in ['vanilla_tit_cql', 'enhanced_tit_cql']:
        args = update_args(args)
        for seed in range(5):
            train(args.env_name, seed, args)

