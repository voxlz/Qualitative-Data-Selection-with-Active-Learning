import glob
import json
import os
from pathlib import Path
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import regex


def get_history_files_from_tests(test_dirs):
    '''For each test directory, loop though the test directory, go into the method folders, and select the last history file. Return a list of history files.'''
    history_files = []
    eval_base = None
    for test_dir in test_dirs:
        for method_dir in os.listdir(test_dir):
            if method_dir == 'eval_base':
                eval_base = get_eval_base(test_dir)
                continue
            if os.path.isdir(os.path.join(test_dir, method_dir)):
                
                # Get the file with the largest number in it's filename in the format "history_epoch_{}.txt" and append it to the list
                all_files = glob.glob(os.path.join(test_dir, method_dir, 'history_*.json'))
                history_file = max(all_files, key=lambda f: int(f.split('_')[-1].strip('.json')))
                history_files.append(history_file)
                
    return history_files, eval_base

def get_optopsy_files_from_logs_old(log_dirs, nbr_epoch):
    '''For each test directory, loop though the test directory, go into the method folders, and select the last history file. Return a list of history files.'''
    eval_files = []
    for log_dir in log_dirs:
        for method_dir in os.listdir(log_dir):
            method_files = []
            for loop_dir in os.listdir(os.path.join(log_dir, method_dir)):
                if os.path.isdir(os.path.join(log_dir, method_dir, loop_dir, 'optopsy_eval')):
                    eval_file = os.path.join(
                        log_dir,
                        method_dir,
                        loop_dir,
                        'optopsy_eval',
                        f"results_epoch_{nbr_epoch}.json",
                    )
                    if os.path.isfile(eval_file):
                        method_files.append(eval_file)
            eval_files.append(method_files)
    return eval_files

def get_optopsy_files_from_logs_new(log_dirs):
    '''For each test directory, loop though the test directory, go into the method folders, and select the last history file. Return a list of history files.'''
    eval_files = []
    method_names = []
    eval_base = None
    for log_dir in log_dirs:
        for method_dir in os.listdir(log_dir):
            if method_dir == 'map_baseline':
                eval_base = glob.glob(os.path.join(log_dir, method_dir, '*', 'results.json'))[0]
                continue
            method_files = []
            method_names.append(method_dir.split('.')[2])
            for loop_dir in os.listdir(os.path.join(log_dir, method_dir)):
                for data_dir in os.listdir(os.path.join(log_dir, method_dir, loop_dir)):
                    eval_file = os.path.join(log_dir, method_dir, loop_dir, data_dir,"results.json")
                    if os.path.isfile(eval_file):
                        method_files.append(eval_file)
            eval_files.append(method_files)
    return eval_files, method_names, eval_base

def plot_image_selection_examples(all_data, processed_data, name: str):
    keys = list(processed_data.keys())

    if not all_data[keys[0]][f'imgs_{name}']:
        return 

    fig = plt.figure(f"{name.capitalize()} Image Examples")
    fig.set_size_inches(20, 12)
    # Image grid of selected images
    cols = [
        f'Loop {col}'
        for col in range(
            0, len(processed_data[keys[0]]['train']['loss']['avg_data'])
        )
    ]
    rows = [f'{row}' for row in keys]

    axes = fig.subplots(nrows=len(rows), ncols=len(cols), squeeze=False)

    for i in range(len(rows)):
        for j in range(len(cols)):
            ax = axes[i, j]

            plt.sca(ax)
            img_path = all_data[keys[i]][f'imgs_{name}'][j]
            file_name = Path(img_path).name
            label = regex.match(f"[0-9]+_({name})+_[0-9]+_.*?(.*).png", file_name).group(2)
            ax.set_xlabel(label)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            img = Image.open(img_path)
            plt.imshow(np.asarray(img))

    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    for ax, row in zip(axes[:,0], rows):
        ax.set_ylabel(row, rotation=0, size='large', labelpad=50)

    fig.tight_layout()
    fig.subplots_adjust(left=0.10, top=0.95)
    word = 'Easiest' if name == 'easy' else 'Hardest'
    fig.suptitle(f'{word} image per selection method', fontsize=16)
    
def get_eval_base(folder):
    path_class = f"{folder}/eval_base/vgg16_baseline.json"
    path_od = f"{folder}/eval_base/od_baseline.json"
    if os.path.exists(path_class):
        with open(path_class, "r") as f: 
            eval_base = json.loads(f.read())
    elif os.path.exists(path_od):
        with open(path_od, "r") as f:
            eval_base = json.loads(f.read())
    else:
        print("Couldn't find file, make sure it has the right name.")
        eval_base = None
    return eval_base