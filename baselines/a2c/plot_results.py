from common.util import get_log_paths
from common.plot import plot_monitors_merged, plot_progress_merged, plot_monitors_individually


def main():
    policy = 'dnc'
    algorithm = 'a2c'
    log_paths = get_log_paths(algorithm, policy)
    title = algorithm + ' training for ' + policy + ' policy'

    plot_monitors_merged(log_paths, title=title)
    plot_progress_merged(log_paths, title=title)
    # plot_monitors_individually(log_paths[-1])


if __name__ == '__main__':
    main()
