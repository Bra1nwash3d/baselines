import os
import shutil
import json


def _basic_path(algorithm_name, policy_name, escapes=2):
    return os.path.abspath('.'+''.join(['/..' for i in range(escapes)])+'/saves/'+algorithm_name+'/'+policy_name)+'/'


def _meta_info_file(algorithm_name, policy_name, escapes=2):
    return _basic_path(algorithm_name, policy_name, escapes) + 'meta.json'


def _model_path(algorithm_name, policy_name, escapes=2):
    return _basic_path(algorithm_name, policy_name, escapes)


def _training_folder_path(algorithm_name, policy_name, training, escapes=2):
    return _basic_path(algorithm_name, policy_name, escapes) + 'training' + str(training) + '/'


def _log_path(algorithm_name, policy_name, training, escapes=2):
    return _training_folder_path(algorithm_name, policy_name, training, escapes) + 'logs/'


def init_next_training(algorithm_name, policy_name, escapes=2):
    meta_info_path = _meta_info_file(algorithm_name, policy_name, escapes)
    info = {}
    try:
        with open(meta_info_path, 'r+') as file:
            info = json.load(file)
            file.close()
    except:
        pass

    # save model in training folder
    if info.get('training_started', 0) > 0:
        path = _basic_path(algorithm_name, policy_name, escapes)
        paste_location = _training_folder_path(algorithm_name, policy_name, info['training_started'], escapes=escapes)
        for name in os.listdir(path):
            if (not name.startswith('training')) and (not '.json' in name):
                os.makedirs(paste_location, exist_ok=True)
                shutil.copy2(path+name, paste_location+name)

    # update meta info, save
    info['training_started'] = info.get('training_started', 0) + 1
    with open(meta_info_path, 'w+') as file:
        json.dump(info, file, indent=2)
        file.close()

    # return necessary paths for training
    model_path = _model_path(algorithm_name, policy_name, escapes=escapes)
    log_path = _log_path(algorithm_name, policy_name, info['training_started'], escapes=escapes)
    return model_path, log_path


def get_log_paths(algorithm_name, policy_name, escapes=2):
    meta_info_path = _meta_info_file(algorithm_name, policy_name, escapes)
    info = {}
    log_paths = []
    try:
        with open(meta_info_path, 'r+') as file:
            info = json.load(file)
            file.close()
    except:
        pass

    for i in range(1, info.get('training_started', 0)+1):
        log_paths.append(_log_path(algorithm_name, policy_name, i, escapes=escapes))

    return log_paths
