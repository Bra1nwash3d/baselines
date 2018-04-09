#!/usr/bin/env python3

import gym
import logging
import os
from baselines.common.atari_wrappers import ClipRewardEnv
from mpi4py import MPI
from baselines.common import set_global_seeds
from baselines import bench
import os.path as osp
from baselines import logger
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.cmd_util import arg_parser
from baselines.ppo1.algorithmic.policies import AlgorithmicFfPolicy
from baselines.ppo1.algorithmic.pposgd_simple import learn
from baselines.common.path_util import init_next_training, set_base_path


def default_env_args():
    return {
        'clipreward': False,
    }


def make_env(env_id, env_args, rank=0, seed=0):
    def _thunk():
        gym.logger.setLevel(logging.WARN)
        env = gym.make(env_id)
        if env_args.get('clipreward', False): env = ClipRewardEnv(env)
        env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)), allow_early_resets=True)
        env.seed(seed+rank)
        return env
    return _thunk


def str_to_policy(policy):
    return {
        'ffalg': AlgorithmicFfPolicy,
    }.get(policy, AlgorithmicFfPolicy)


def train(env_id, env_args, policy, policy_args, num_cpu, num_timesteps, seed):
    import baselines.common.tf_util as U

    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.make_session(num_cpu=num_cpu)
    sess.__enter__()
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)

    env = make_env(env_id, env_args, rank=rank, seed=workerseed)()

    """
    env = make_atari(env_id)
    env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), str(rank)))
    env.seed(workerseed)
    env = wrap_deepmind(env)
    env.seed(workerseed)
    """

    learn(env=env,
          env_args=env_args,
          policy_fn=str_to_policy(policy),
          policy_args=policy_args,
          max_timesteps=int(num_timesteps * 1.1),
          timesteps_per_actorbatch=256,
          clip_param=0.2,
          entcoeff=0.01,
          optim_epochs=4,
          optim_stepsize=1e-3,
          optim_batchsize=64,
          gamma=0.99,
          lam=0.95,
          schedule='linear')
    env.close()


def main():
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', default='Copy-v0')
    # parser.add_argument('--env', help='environment ID', default='RepeatCopy-v0')
    # parser.add_argument('--env', help='environment ID', default='Reverse-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=8)
    parser.add_argument('--num-timesteps', type=int, default=int(10e5))
    parser.add_argument('--policy', help='Policy architecture', choices=['ffalg'], default='ffalg')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    args = parser.parse_args()
    policy_args = {
        'hidden': 20,
    }
    env_args = default_env_args()
    model_path, log_path, policy_args, env_args = init_next_training('ppo1', args.policy, args.env, policy_args, env_args)
    logger.configure(dir=log_path)
    train(env_id=args.env,
          env_args=env_args,
          policy=args.policy,
          policy_args=policy_args,
          num_cpu=4,
          num_timesteps=args.num_timesteps,
          seed=args.seed)


if __name__ == '__main__':
    main()
