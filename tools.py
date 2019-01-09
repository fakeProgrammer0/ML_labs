'''Export the dataset directory's path'''

import os
import requests
import matplotlib.pyplot as plt

PROJECT_ABS_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_ABS_PATH, 'dataset')


def download_dataset(dataset_url):
    '''A Helper function to download dataset

    Parameters
    ----------
    dataset_url: str
        The url of the dataset

    Return
    ------
    dataset_file_path: str
        The absolute path of the dataset

    '''
    dataset_filename = dataset_url[dataset_url.rfind('/') + 1:]
    dataset_file_path = os.path.join(DATASET_DIR, dataset_filename)

    # load dataset online
    if not os.path.exists(dataset_file_path):
        if not os.path.exists(DATASET_DIR):
            os.makedirs(DATASET_DIR)

        r = requests.get(dataset_url)
        with open(dataset_file_path, 'w', encoding='utf-8') as f:
            f.write(r.text)

    return dataset_file_path


def plot_losses_graph(losses_dict,
                      title="losses graph",
                      xlabel="epoch",
                      ylabel='loss',
                      params_dict=None,
                      params_notation_pos_width_perc=0.8,
                      params_notation_pos_height_perc=0.8):
    '''
    A helper function used to draw the losses graph.

    Parameters
    ----------
    losses_dict : dict with (key, values) in the form (losses_label, losses_data)
        A dict containing losses information.
            losses_label : str
                A label indicating the information about the loss.
            losses_data : list
                A list consisting of loss data.

    title : str
        The title of the losses graph.

    xlabel, ylabel : str
        The label of the losses graph.

    params_dict : dict with (key, values) in the form (param_name, param_value)
        param_name : str
            Name of the parameter
        param_value : str
            Value of the parameter

    params_notation_pos_width_perc, params_notation_pos_height_perc : float, range in [0, 1.0]
        Used to adjust the posisiton of the parameter notation
    
    '''
    colors = ['r', 'b', 'k', 'g', 'c', 'm', 'y']

    plt.figure(figsize=(16, 9))
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    max_width, max_height = 0, 0

    for i, losses_label in enumerate(losses_dict):
        losses_data = losses_dict.get(losses_label)
        plt.plot(
            losses_data,
            '-',
            color=colors[i % len(colors)],
            label=losses_label)

        max_width = max(max_width, len(losses_data))
        max_height = max(max_height, max(losses_data))

    if params_dict is not None:
        param_str = ''
        for param_name in params_dict:
            param_str += '\n' + param_name + ' : ' + str(
                params_dict[param_name])
        plt.text(
            max_width * params_notation_pos_width_perc,
            max_height * params_notation_pos_height_perc,
            param_str,
            fontsize=12)

    plt.legend()
    plt.show()


def execute_procedure(f, desc=''):
    print(f'Running [{desc}] in the background...\nPlease wait...')
    f()
    print('Done!')
