#!/usr/bin/env python
import sys, time
import argparse
from baselines import bench, logger
from baselines.common.plot_util import init_next_training

from baselines.ppo2.ppo2_Predictive2 import learn
from baselines.ppo2.policies_Predictive import DncPolicyPredictive2
from baselines.a2c.run_atari_Predictive2 import default_env_args, make_env


def str_to_policy(policy):
    return {
        'dncpred2': DncPolicyPredictive2,
    }.get(policy, DncPolicyPredictive2)


def train(env_id, env_args, num_timesteps, seed, policy, policy_args, save_path, num_cpu):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
    from baselines.common.vec_env.vec_frame_stack import VecFrameStack
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

    env = SubprocVecEnv([make_env(env_id, env_args, rank=i, seed=seed) for i in range(num_cpu)])
    set_global_seeds(seed)
    env = VecFrameStack(env, 4)
    learn(policy=str_to_policy(policy),
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
    parser.add_argument('--policy', help='Policy architecture', choices=['dncpred2'], default='dncpred2')
    parser.add_argument('--num-timesteps', type=int, default=int(10e8))
    args = parser.parse_args()
    env_args = default_env_args('pixel')
    # env_args = default_env_args('classic')
    policy_args = {
        'memory_size': 4,  # 8
        'word_size': 16,  # 8
        'num_read_heads': 1,
        'num_write_heads': 1,
        'clip_value': 200000,
        'nlstm': 256,  # 128, 16, DNC controller
        'ndncout': 128,  # 128, 16, DNC output size
        'nobsdesc': 512,  # 512, 8, size of the observation-describing vector that enters the DNC, and is used for future prediction
        'nobsdesch': 256,  # 256, 8, hidden layer before nobsdesc
    }
    model_path, log_path, policy_args, env_args = init_next_training('ppo2', args.policy, args.env, policy_args, env_args)
    logger.configure(dir=log_path)
    t0 = time.time()

    train(args.env,
          env_args,
          num_timesteps=args.num_timesteps,
          seed=args.seed,
          policy=args.policy,
          policy_args=policy_args,
          save_path=model_path,
          num_cpu=16)
    logger.info("Training time: \t\t%.2f" % (time.time()-t0))

if __name__ == '__main__':
    main()
