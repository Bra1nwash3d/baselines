#!/usr/bin/env python
import gym
import logging
import os
import time

from baselines import bench
from baselines import logger
from baselines.a2c.a2c import learn
from baselines.a2c.policies import CnnPolicy, LstmPolicy, LnLstmPolicy, DncPolicy
from baselines.common import set_global_seeds
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.path_util import init_next_training
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv


def str_to_policy(policy):
    return {
        'cnn': CnnPolicy,
        'lstm': LstmPolicy,
        'lnlstm': LnLstmPolicy,
        'dnc': DncPolicy,
    }.get(policy, DncPolicy)


def train(env_id, num_timesteps, seed, policy, lrschedule, num_cpu, save_path, nsteps=1):
    def make_env(rank):
        def _thunk():
            env = make_atari(env_id)
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            gym.logger.setLevel(logging.WARN)
            return wrap_deepmind(env)
        return _thunk
    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    learn(str_to_policy(policy),
          env,
          seed,
          nsteps=nsteps,
          total_timesteps=int(num_timesteps * 1.1),
          lrschedule=lrschedule,
          save_path=save_path)
    env.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'dnc'], default='dnc')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    parser.add_argument('--num-timesteps', type=int, default=int(1*10e5))
    args = parser.parse_args()
    t0 = time.time()
    model_path, log_path = init_next_training('a2c', args.policy)
    logger.configure(dir=log_path)
    train(args.env,
          num_timesteps=args.num_timesteps,
          seed=args.seed,
          policy=args.policy,
          lrschedule=args.lrschedule,
          num_cpu=8,
          nsteps=10,
          save_path=model_path)
    logger.info("Training time: \t\t%.2f" % (time.time()-t0))


if __name__ == '__main__':
    main()
