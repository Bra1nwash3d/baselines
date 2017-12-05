from baselines.common.path_util import get_log_paths
from baselines.common.plot_util import plot_monitors_merged, plot_progress_merged, plot_monitors_individually


def main():
    policy = 'dnc'
    algorithm = 'a2c'
    env_id = 'BreakoutNoFrameskip-v4'
    log_paths = get_log_paths(algorithm, policy, env_id)
    title = algorithm + ' training for ' + policy + ' policy'

    if len(log_paths) <= 0:
        print('Nothing to plot!')
        return

    plot_monitors_merged(log_paths, title=title)
    plot_progress_merged(log_paths, title=title)
    # plot_monitors_individually(log_paths[-1])


if __name__ == '__main__':
    main()
