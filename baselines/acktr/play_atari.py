#!/usr/bin/env python
from baselines.a2c.a2c import play
from baselines.a2c.run_atari import str_to_policy
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.path_util import get_model_path


def play_atari(env_id, seed, policy, save_path, num_episodes):
    env = make_atari(env_id)
    env = wrap_deepmind(env)
    env.seed(seed)
    play(str_to_policy(policy), env, seed, nep=num_episodes, save_path=save_path, save_name='model')
    env.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'dnc'], default='dnc')
    parser.add_argument('--num-episodes', type=int, default=5)
    parser.add_argument('--training', type=int, default=-1)
    args = parser.parse_args()
    model_path = get_model_path('a2c', args.policy, args.env, training=args.training)
    if model_path:
        play_atari(args.env,
                   seed=args.seed,
                   policy=args.policy,
                   save_path=model_path,
                   num_episodes=args.num_episodes)


if __name__ == '__main__':
    main()
