#!/usr/bin/env python3
import sys
from baselines import logger
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.path_util import init_next_training
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy, DncPolicy
import multiprocessing
import tensorflow as tf


def train(env_id, num_timesteps, seed, policy, policy_args, env_args, save_path):

    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    tf.Session(config=config).__enter__()

    env = VecFrameStack(make_atari_env(env_id, 8, seed), 4)
    policy = {'cnn' : CnnPolicy,
              'lstm' : LstmPolicy,
              'lnlstm' : LnLstmPolicy,
              'dnc' : DncPolicy}[policy]
    ppo2.learn(policy=policy,
               policy_args=policy_args,
               env=env,
               env_args=env_args,
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
    parser = atari_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    args = parser.parse_args()

    policy_args = {
        'nlstm': 256,
    }
    env_args = {}
    model_path, log_path, policy_args, env_args = init_next_training('ppo2', args.policy, args.env, policy_args, env_args)
    logger.configure(dir=log_path)
    train(args.env,
          num_timesteps=args.num_timesteps,
          seed=args.seed,
          policy=args.policy,
          policy_args=policy_args,
          env_args=env_args,
          save_path=model_path)


if __name__ == '__main__':
    main()
