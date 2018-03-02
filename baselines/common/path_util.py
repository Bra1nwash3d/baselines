import os
import shutil
import json
import csv

base_path = False  # relative path if False


def set_base_path(path):
    global base_path
    base_path = path


def _basic_path(algorithm_name, policy_name, env_id, escapes=2):
    if base_path:
        return base_path+algorithm_name+'/'+policy_name+'/'+env_id+'/'
    return os.path.abspath('.'+''.join(['/..' for i in range(escapes)])+
                           '/saves/'+algorithm_name+'/'+policy_name)+'/'+env_id+'/'


def _meta_info_file(algorithm_name, policy_name, env_id, escapes=2):
    return _basic_path(algorithm_name, policy_name, env_id, escapes) + 'meta.json'


def _model_path(algorithm_name, policy_name, env_id, escapes=2):
    return _basic_path(algorithm_name, policy_name, env_id, escapes)


def _training_folder_path(algorithm_name, policy_name, env_id, training, escapes=2):
    return _basic_path(algorithm_name, policy_name, env_id, escapes) + 'training' + str(training) + '/'


def _log_path(algorithm_name, policy_name, env_id, training, escapes=2):
    return _training_folder_path(algorithm_name, policy_name, env_id, training, escapes) + 'logs/'


def init_next_training_args(algorithm_name, policy_name, env_id, policy_args, env_args, escapes=2):
    meta_info_path = _meta_info_file(algorithm_name, policy_name, env_id, escapes)
    info = {}
    try:
        with open(meta_info_path, 'r+') as file:
            info = json.load(file)
            file.close()
    except:
        os.makedirs(meta_info_path, exist_ok=True)
        os.rmdir(meta_info_path)
        pass

    trained_timesteps = 0
    if info.get('training_started', 0) > 0:
        # save model in training folder
        path = _basic_path(algorithm_name, policy_name, env_id, escapes)
        paste_location = _training_folder_path(algorithm_name, policy_name, env_id, info['training_started'], escapes=escapes)
        for name in os.listdir(path):
            if (not name.startswith('training')) and (not '.json' in name):
                os.makedirs(paste_location, exist_ok=True)
                shutil.copy2(path+name, paste_location+name)
        # see how many timesteps were already trained
        log_path = _log_path(algorithm_name, policy_name, env_id, info['training_started'])
        with open(log_path+'progress.csv') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            last_row = {}
            for row in reader:
                last_row = row
            trained_timesteps = last_row.get('total_timesteps', 0)

    # load policy args if available
    if info.get('policy_args', False):
        policy_args = info.get('policy_args')
    else:
        info['policy_args'] = policy_args

    # load environment args if available
    if info.get('env_args', False):
        env_args = info.get('env_args')
    else:
        info['env_args'] = env_args

    # update training info, save
    info['training_started'] = info.get('training_started', 0) + 1
    with open(meta_info_path, 'w+') as file:
        json.dump(info, file, indent=2)
        file.close()

    # return necessary paths for training
    model_path = _model_path(algorithm_name, policy_name, env_id, escapes=escapes)
    log_path = _log_path(algorithm_name, policy_name, env_id, info['training_started'], escapes=escapes)
    return {
        'model_path': model_path,
        'log_path': log_path,
        'policy_args': policy_args,
        'env_args': env_args,
        'trained_timesteps': trained_timesteps,
    }


def init_next_training(algorithm_name, policy_name, env_id, policy_args, env_args, escapes=2):
    args = init_next_training_args(algorithm_name, policy_name, env_id, policy_args, env_args, escapes=escapes)
    return args['model_path'], args['log_path'], args['policy_args'], args['env_args']


def get_log_paths(algorithm_name, policy_name, env_id, escapes=2):
    meta_info_path = _meta_info_file(algorithm_name, policy_name, env_id, escapes)
    info = {}
    log_paths = []
    try:
        with open(meta_info_path, 'r+') as file:
            info = json.load(file)
            file.close()
    except:
        pass

    for i in range(1, info.get('training_started', 0)+1):
        log_paths.append(_log_path(algorithm_name, policy_name, env_id, i, escapes=escapes))

    return log_paths


def get_model_path_and_args(algorithm_name, policy_name, env_id, escapes=2, training=-1):
    path = '.'
    if training > 0:
        # model of specific training
        path = _training_folder_path(algorithm_name, policy_name, env_id, training, escapes=2)
    else:
        # most recent model for training <= 0
        path = _model_path(algorithm_name, policy_name, env_id, escapes=escapes)
    if not os.path.exists(path):
        print('Model does not exist!')
        print(path)
        return False, {}, {}

    meta_info_path = _meta_info_file(algorithm_name, policy_name, env_id, escapes)
    info = {}
    try:
        with open(meta_info_path, 'r+') as file:
            info = json.load(file)
            file.close()
    except:
        print("Meta info not available, can't find policy args")
        return False, {}, {}
    return path, info.get('policy_args', {}), info.get('env_args', {})
