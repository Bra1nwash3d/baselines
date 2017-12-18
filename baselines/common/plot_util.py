import pandas as pd
import os
import re
import matplotlib.pyplot as plt


def col_name_to_str(col_name):
    return {
        'r': 'Total episode reward',
        'l': 'Episode length',
        't': 'Time since start',
        'policy_entropy': 'Policy entropy',
        'policy_loss': 'Policy loss',
        'fps': 'Frames/s',
        'value_loss': 'Value Loss',
        'total_timesteps': 'Total timesteps',
        'nupdates': '#Updates',
        'explained_variance': 'Explained variance'
    }.get(col_name, col_name)


def plot_monitors_individually(log_path, rolling_window=20):
    dfs = []
    for fn in [f for f in os.listdir(log_path) if re.match(r'[0-9]+.monitor.csv', f)]:
        dfs.append((int(fn.split('.',1)[0]), pd.read_csv(log_path+fn, comment='#')))
    col_names = [a for a in dfs[0][1].axes[1]]
    dfs = sorted(dfs, key=lambda d: d[0])

    for i in range(len(col_names)-1):
        # not plotting training time
        col_name = col_names[i]
        fig = plt.figure(1)
        fig.suptitle(col_name_to_str(col_name))
        for j in range(len(dfs)):
            df = dfs[j][1]
            plt.subplot(len(dfs), 1, j+1)
            plt.plot(df[col_name], 'bo')
            plt.plot(df[col_name].rolling(window=rolling_window).mean(), 'r', linewidth=3)
            plt.ylabel('Monitor-'+str(dfs[j][0]))
        plt.show()


def plot_monitors_merged(log_paths, rolling_window=200, sort_by='t', title='Training', show=True, save_path=False):
    # merge monitors of training session, concat training sessions, correct time spent
    dfs = []
    last_value = 0
    for i in range(len(log_paths)):
        try:
            local_dfs = []
            log_path = log_paths[i]
            for fn in [f for f in os.listdir(log_path) if re.match(r'[0-9]+.monitor.csv', f)]:
                local_dfs.append(pd.read_csv(log_path + fn, comment='#'))
            df = pd.concat(local_dfs, ignore_index=True)
            df = df.sort_values(by=sort_by)
            df[sort_by] += last_value
            last_value = df[sort_by][len(df)-1]
            dfs.append(df)
        except:
            # may fail if the model is training and created a yet empty last set of logs
            break

    col_names = [a for a in dfs[0].axes[1]]
    df = pd.concat(dfs, ignore_index=True)

    fig = plt.figure(1)
    col_names = col_names[:]
    fig.suptitle(title)
    for i in range(len(col_names)):
        # not plotting training time
        col_name = col_names[i]
        plt.subplot(len(col_names), 1, i+1)
        plt.scatter(range(len(df)), df[col_name], s=[0.2 for _ in range(len(df))])
        plt.plot(range(len(df)), df[col_name].rolling(window=rolling_window).mean(), 'r', linewidth=3)
        plt.ylabel(col_name_to_str(col_name))
    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path)


def plot_progress_merged(log_paths, rolling_window=200, title='Training', show=True, save_path=False):
    # concat progress of training sessions, correct total timesteps and update count spent
    dfs = []
    last_total_timesteps = 0
    last_nupdates = 0
    for i in range(len(log_paths)):
        try:
            local_dfs = []
            log_path = log_paths[i]
            df = pd.read_csv(log_path+'progress.csv', comment='#')
            df['total_timesteps'] += last_total_timesteps
            last_total_timesteps = df['total_timesteps'][len(df)-1]
            df['nupdates'] += last_nupdates
            last_nupdates = df['nupdates'][len(df)-1]
            dfs.append(df)
        except:
            # may fail if the model is training and created a yet empty last set of logs
            break

    col_names = [a for a in dfs[0].axes[1]]
    df = pd.concat(dfs, ignore_index=True)

    fig = plt.figure(1)
    col_names = col_names[:]
    fig.suptitle(title)
    for i in range(len(col_names)):
        # not plotting training time
        col_name = col_names[i]
        plt.subplot(len(col_names), 1, i+1)
        plt.scatter(range(len(df)), df[col_name], s=[0.6 for _ in range(len(df))])
        # plt.plot(range(len(df)), df[col_name].rolling(window=rolling_window).mean(), 'r', linewidth=3)
        plt.ylabel(col_name_to_str(col_name))
    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path)
