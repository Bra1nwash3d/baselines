#!/usr/bin/env python3

from baselines import logger
from baselines.common.path_util import init_next_training
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.a2c.a2c import learn
from baselines.a2c.policies import CnnPolicy, LstmPolicy, LnLstmPolicy, DncPolicy


def str_to_policy(policy):
    return {
        'cnn': CnnPolicy,
        'lstm': LstmPolicy,
        'lnlstm': LnLstmPolicy,
        'dnc': DncPolicy,
    }.get(policy, CnnPolicy)


def train(env_id, env_args, policy_args, num_timesteps, seed, policy, lrschedule, num_env, model_path=False):
    policy_fn = str_to_policy(policy)
    env = VecFrameStack(make_atari_env(env_id, num_env, seed), 4)
    learn(policy_fn, policy_args, env, env_args, seed, total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule,
          save_path=model_path)
    env.close()


def main():
    parser = atari_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'dnc'], default='cnn')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    args = parser.parse_args()
    policy_args = {}
    env_args = {}
    model_path, log_path, policy_args, env_args = init_next_training('a2c', args.policy, args.env, policy_args,
                                                                     env_args)
    logger.configure(dir=log_path)
    train(args.env,
          env_args=env_args,
          policy_args=policy_args,
          num_timesteps=args.num_timesteps,
          seed=args.seed,
          policy=args.policy,
          lrschedule=args.lrschedule,
          num_env=16,
          model_path=model_path)


if __name__ == '__main__':
    main()
