#!/usr/bin/env python
import gym
import logging
import os
import time

from baselines import bench
from baselines import logger
from baselines.a2c.algorithmic.a2c import learn
from baselines.a2c.algorithmic.policies import DncPolicy
from baselines.common import set_global_seeds
from baselines.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, ScaledFloatFrame, \
    ClipRewardEnv, FrameStack, WarpFrame
from baselines.common.path_util import init_next_training
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv


def default_env_args():
    return {
        'noopreset': -1,
        'maxandskip': 1,
        'episodiclife': False,
        'warpframe': False,
        'scaledfloat': False,
        'clipreward': False,
        'framestack': 1,
    }


def make_env(env_id, env_args, rank=0, seed=0):
    def _thunk():
        gym.logger.setLevel(logging.WARN)
        env = gym.make(env_id)
        if env_args.get('noopreset', 0) > 0: env = NoopResetEnv(env, noop_max=env_args.get('noopreset'))
        if env_args.get('maxandskip', 1) > 1: env = MaxAndSkipEnv(env, skip=env_args.get('maxandskip'))
        if env_args.get('episodiclife', False): env = EpisodicLifeEnv(env)
        if env_args.get('warpframe', False): env = WarpFrame(env)
        if env_args.get('scaledfloat', False): env = ScaledFloatFrame(env)
        if env_args.get('clipreward', False): env = ClipRewardEnv(env)
        if env_args.get('framestack', 1) > 1: env = FrameStack(env, env_args.get('framestack'))
        env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)), allow_early_resets=True)
        env.seed(seed+rank)
        return env
    return _thunk


def str_to_policy(policy):
    return {
        'dncalg': DncPolicy,
    }.get(policy, DncPolicy)


def train(policy_args, env_id, env_args, num_timesteps, seed, policy, lrschedule, num_cpu, save_path, nsteps=1):
    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(env_id, env_args, rank=i, seed=seed) for i in range(num_cpu)])

    learn(str_to_policy(policy),
          policy_args,
          env,
          env_args,
          seed,
          nsteps=nsteps,
          total_timesteps=int(num_timesteps * 1.1),
          lrschedule=lrschedule,
          save_path=save_path)
    env.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Copy-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--policy', help='Policy architecture', choices=['dncalg'], default='dncalg')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    parser.add_argument('--num-timesteps', type=int, default=int(10e5))
    args = parser.parse_args()
    t0 = time.time()
    policy_args = {
        'memory_size': 16,
        'word_size': 16,
        'num_read_heads': 2,
        'num_write_heads': 1,
        'clip_value': 200000,
        'nlstm': 16,
    }

    env_args = default_env_args()
    model_path, log_path, policy_args, env_args = init_next_training('a2c', args.policy, args.env, policy_args, env_args)
    logger.configure(dir=log_path)
    train(policy_args,
          args.env,
          env_args,
          num_timesteps=args.num_timesteps,
          seed=args.seed,
          policy=args.policy,
          lrschedule=args.lrschedule,
          num_cpu=16,
          nsteps=5,
          save_path=model_path)
    logger.info("Training time: \t\t%.2f" % (time.time()-t0))


if __name__ == '__main__':
    main()
