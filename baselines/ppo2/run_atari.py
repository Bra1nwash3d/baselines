#!/usr/bin/env python
import sys, time
import argparse
from baselines import bench, logger
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy, DncPolicy
from baselines.common.path_util import init_next_training


def str_to_policy(policy):
    return {
        'cnn': CnnPolicy,
        'lstm': LstmPolicy,
        'lnlstm': LnLstmPolicy,
        'dnc': DncPolicy,
    }.get(policy, DncPolicy)


def train(env_id, num_timesteps, seed, policy, policy_args, save_path, num_cpu):
    from baselines.common import set_global_seeds
    from baselines.common.atari_wrappers import make_atari, wrap_deepmind
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
    from baselines.common.vec_env.vec_frame_stack import VecFrameStack
    from baselines.ppo2 import ppo2
    import gym
    import logging
    import multiprocessing
    import os.path as osp
    import tensorflow as tf
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    gym.logger.setLevel(logging.WARN)
    tf.Session(config=config).__enter__()

    def make_env(rank):
        def env_fn():
            env = make_atari(env_id)
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), str(rank)))
            return wrap_deepmind(env)
        return env_fn
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    set_global_seeds(seed)
    env = VecFrameStack(env, 4)
    ppo2.learn(policy=str_to_policy(policy),
               policy_args=policy_args,
               env=env,
               nsteps=128,
               nminibatches=4,
               lam=0.95,
               gamma=0.99,
               noptepochs=4,
               log_interval=1,
               ent_coef=.01,
               lr=lambda f : f * 2.5e-4,
               cliprange=lambda f : f * 0.1,
               total_timesteps=int(num_timesteps * 1.1),
               save_path=save_path)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'dnc'], default='lstm')
    parser.add_argument('--num-timesteps', type=int, default=int(10e8))
    args = parser.parse_args()
    logger.configure()

    """
    # Use this for DNC policy.
    # Also requires a small code change in ppo2.py to use the dnc state subset for training
    policy_args = {
        'memory_size': 8,
        'word_size': 8,
        'num_read_heads': 1,
        'num_write_heads': 1,
        'clip_value': 200000,
        'nlstm': 64,
    }"""
    policy_args = {
        'nlstm': 256,
    }

    model_path, log_path, policy_args, _ = init_next_training('ppo2', args.policy, args.env, policy_args, {})
    logger.configure(dir=log_path)
    t0 = time.time()
    train(args.env,
          num_timesteps=args.num_timesteps,
          seed=args.seed,
          policy=args.policy,
          policy_args=policy_args,
          save_path=model_path,
          num_cpu=16)
    logger.info("Training time: \t\t%.2f" % (time.time()-t0))

if __name__ == '__main__':
    main()
