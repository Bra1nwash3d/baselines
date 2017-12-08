#!/usr/bin/env python
import os, logging, gym, time
from baselines import logger
from baselines.common import set_global_seeds
from baselines import bench
from baselines.acktr.acktr_disc import learn
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.acktr.policies import CnnPolicy, DncPolicy
from baselines.common.path_util import init_next_training


def str_to_policy(policy):
    return {
        'cnn': CnnPolicy,
        'dnc': DncPolicy,
    }.get(policy, DncPolicy)


def train(policy_args, env_id, num_timesteps, seed, policy, num_cpu, save_path):
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
          policy_args,
          env,
          seed,
          total_timesteps=int(num_timesteps * 1.1),
          nprocs=num_cpu,
          save_path=save_path)
    env.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'dnc'], default='dnc')
    parser.add_argument('--num-timesteps', type=int, default=int(1*10e5))
    args = parser.parse_args()
    t0 = time.time()
    policy_args = {
        'memory_size': 16,
        'word_size': 16,
        'num_read_heads': 2,
        'num_write_heads': 1,
        'clip_value': 200000,
        'nlstm': 64,
    }
    model_path, log_path, policy_args = init_next_training('acktr', args.policy, args.env, policy_args)
    logger.configure(dir=log_path)
    train(policy_args,
          args.env,
          num_timesteps=args.num_timesteps,
          seed=args.seed,
          policy=args.policy,
          num_cpu=16,
          save_path=model_path)
    logger.info("Training time: \t\t%.2f" % (time.time()-t0))


if __name__ == '__main__':
    main()
