#!/usr/bin/env python3
import os
import gym
import logging
from baselines import bench, logger
from baselines.common.path_util import init_next_training
from baselines.common.cmd_util import arg_parser
from baselines.common.atari_wrappers import ClipRewardEnv
from baselines.a2c.algorithmic.a2c import learn
from baselines.a2c.algorithmic.policies import AlgorithmicDncPolicy
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv


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
        'dncalg': AlgorithmicDncPolicy,
    }.get(policy, AlgorithmicDncPolicy)


def train(env_id, env_args, policy_args, num_timesteps, seed, policy, lrschedule, num_env, model_path=False):
    policy_fn = str_to_policy(policy)
    env = SubprocVecEnv([make_env(env_id, env_args, rank=i, seed=seed) for i in range(num_env)])
    learn(policy_fn, policy_args,
          env, env_args,
          seed,
          total_timesteps=int(num_timesteps * 1.1),
          lrschedule=lrschedule,
          lr=5e-4,
          nsteps=5,
          log_interval=500,  # 500
          save_path=model_path)
    env.close()


def main():
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', default='DuplicatedInput-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=8)
    parser.add_argument('--num-timesteps', type=int, default=int(10e5))
    parser.add_argument('--policy', help='Policy architecture', choices=['dncalg'], default='dncalg')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    args = parser.parse_args()
    policy_args = {
        'memory_size': 16,
        'word_size': 16,
        'num_read_heads': 1,
        'num_write_heads': 1,
        'clip_value': 200000,
        'num_controller_lstm': 64,
        'num_dnc_out': 32,
    }
    env_args = default_env_args()
    model_path, log_path, policy_args, env_args = init_next_training('a2c_new', args.policy, args.env, policy_args,
                                                                     env_args)
    logger.configure(dir=log_path)
    train(args.env,
          env_args=env_args,
          policy_args=policy_args,
          num_timesteps=args.num_timesteps,
          seed=args.seed,
          policy=args.policy,
          lrschedule=args.lrschedule,
          num_env=4,
          model_path=model_path)


if __name__ == '__main__':
    main()
