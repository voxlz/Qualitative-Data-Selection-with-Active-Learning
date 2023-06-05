import glob
import json
import math
import os
from collections import defaultdict
from itertools import accumulate
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import regex
from active_learning.active_learning_utils import Prio
from utils.latex_utils import print_export_auc_table, print_export_ttest_file
from utils.plot_utils import plot_image_selection_examples
from itertools import product

def plot_history_files(
    history_rel_paths,
    skip_first_eval             = True,
    eval_baseline       : dict  = None,
    baseline            : dict  = None,
    export_latex                = False,
    save_avg                    = False,
    separate_budgets            = False,
    show_textbox                = False,
    separate_pivots             = False,
    separate_accumulate         = False,
    use_standard_deviation      = False,
    suptitle                    = None,
    plot_epochs                 = False
):
    '''
        Plots the history files in the given list of paths. Produces a plot for each metric, for train and eval.
        If multiple files have the same selection method and sample size, they are plotted together as average + range.
        Supports different sample sizes.

        Budget-schedule is calculated from the history files, either from *all of them* or the *last one repeating* if there are less files than epochs.
    '''

    avg_data_list    = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Read all files and save the data
    all_data, metric_keys, info = read_data_from_files(history_rel_paths, skip_first_eval, separate_budgets, separate_pivots, eval_baseline, separate_accumulate)

    # Process the data
    processed_data = preprocess_plotting_data(eval_baseline, all_data, metric_keys, info)

    # Sort the data based on enum value
    myKeys = list(processed_data.keys())
    myKeys.sort(key=lambda key: Prio[key.split('-')[0]].value)
    processed_data = {i: processed_data[i] for i in myKeys}

    # Plot the saved and processed data
    plot_processed_data(baseline, show_textbox, metric_keys, processed_data, use_standard_deviation, suptitle, info, plot_epochs)

    # Export area under curve table
    if export_latex:
        print_export_auc_table(processed_data, metric_keys)
        print_export_ttest_file(processed_data, metric_keys)

    # Save the average data as a baseline
    if save_avg:
        os.makedirs('baseline', exist_ok=True)
        path = os.path.join('baseline', 'XXX.json')
        with open(path, 'w') as f:
            f.write(json.dumps(avg_data_list))

    # Show image examples
    plot_image_selection_examples(all_data, processed_data, 'easy')
    plot_image_selection_examples(all_data, processed_data, 'hard')
    plot_image_selection_examples(all_data, processed_data, 'hard_selected')

    plt.show()

    print()

def read_data_from_files(history_rel_paths, skip_first_eval, separate_budgets, separate_pivots, eval_baseline, separate_accumulate):
    ''' Reads the data from all the given history files. '''

    all_data = defaultdict(lambda: {
        'train': defaultdict(list),
        'eval': defaultdict(list),
        'budget': None,
        'imgs_easy': [],
        'imgs_hard': [],
        'imgs_hard_selected': [],
        'nbr_epochs': []
    })
    metric_keys = []
    first_info = None

    for history_rel_path in history_rel_paths:
        with open(history_rel_path, 'r') as f:
            file = json.loads(f.read())

        all_data, metric_keys, first_info = read_from_file(
            skip_first_eval, 
            separate_budgets, 
            separate_pivots, 
            eval_baseline, 
            separate_accumulate, 
            all_data, first_info, 
            history_rel_path, 
            file
        )

    return all_data, metric_keys, first_info

def read_from_file(skip_first_eval, separate_budgets, separate_pivots, eval_baseline, separate_accumulate, all_data, first_info, history_rel_path, file):
    if first_info is None:
        first_info = file['info']

    # Get keys for metrics
    metric_keys = list(file['train'][0]['history'].keys())
    eval_metric_keys = file['eval'][0].keys()
    assert metric_keys, f'No metrics found in file: {history_rel_path}'

    # Ensure that the metrics are the same in train and eval
    if metric_keys != eval_metric_keys:
        metric_keys = [metric for metric in metric_keys if metric in eval_metric_keys]

    # Loop through all the history files and save 'info' and current epochs train and eval data as a list of dicts
    parent_path = Path(history_rel_path).parent
    history_files = glob.glob(os.path.join(parent_path, 'history_*.json'))
    trainings = get_trainings(history_files)

    prio_func = get_prio_func_name(separate_budgets, separate_pivots, separate_accumulate, file)

    all_data = get_images(all_data, prio_func, parent_path)

    # For each dataset (train, eval) and metric, save the data
    datasets = ('train', 'eval')
    for data_set, metric in product(datasets, metric_keys):
        if data_set == 'train':
            # Save best value for each loop
            values = []
            nbr_epochs = []
            for training in trainings:
                patience = get_patience(training)
                values.append(training['train'][metric][patience])
                nbr_epochs.append(len(training['train'][metric]) + patience + 1)

        else:
            # Save metric values for each loop
            evaluation = file['eval'][1:] if skip_first_eval else file['eval']
            values = [eval_loop[metric] for eval_loop in evaluation]

            # Insert the evaluation baseline if it is given
            if eval_baseline:
                values.insert(0, eval_baseline[metric])

        all_data[prio_func][data_set][metric].append(values)
    all_data[prio_func]['nbr_epochs'].append(nbr_epochs)

    # Load all related epoch files and figure out the budget schedule
    path = Path(history_rel_path).parent
    epoch_files = glob.glob(os.path.join(path, 'history_*.json'))
    budget_schedule = []
    for file in epoch_files:
        with open(file, 'r') as f:
            obj_str = f.read()
            file = json.loads(obj_str)
            budget_schedule.append(file['info']['sample_count'])

        # Set budget
        # al_loop_count = len(all_data[prio_func]['train'][metric_keys[0]][0])
        # all_epoch_files = len(budget_schedule) == al_loop_count
        # aaaaa = np.repeat(budget_schedule, 2)

    all_data[prio_func]['budget'] = budget_schedule
    return all_data,metric_keys,first_info

def get_prio_func_name(separate_budgets, separate_pivots, separate_accumulate, file):
    prio_func = file['info']['prio_func']
    
    # Separate budgets from each other by giving them unique names
    if separate_budgets:
        prio_func += f"-{file['info']['sample_count']}"

    # Separate pivot data from each other by giving them unique names
    if separate_pivots and 'pivot' in file['info']:
        pivot = file['info']['pivot']
        prio_func += '-p%.2f' % pivot

    # Separate accumulate data from each other by giving them unique names
    if separate_accumulate and 'accumulate' in file['info']:
        acc = file['info']['accumulate']
        prio_func += f'-{acc}'
    return prio_func

def get_images(all_data, prio_func, parent_path):
    img_files = glob.glob(os.path.join(parent_path, '*.png'))
    img_files.extend(glob.glob(os.path.join(parent_path, '*.jpg')))

    easy_imgs = list(filter(lambda file_name: 'easy' in file_name,                                  img_files))
    sele_imgs = list(filter(lambda file_name: 'hard_selected' in file_name,                         img_files))
    hard_imgs = list(filter(lambda file_name: regex.match(r".*hard_[0-9]+.*", file_name) != None,   img_files))

    if prio_func not in all_data:
        all_data[prio_func]['imgs_easy'] = easy_imgs
        all_data[prio_func]['imgs_hard'] = hard_imgs
        all_data[prio_func]['imgs_hard_selected'] = sele_imgs
    
    return all_data

def get_trainings(history_files):
    trainings = []
    history_files = sorted(history_files, key=lambda x: int(regex.match(r".*history_([0-9]+).json", x).group(1)))

    for history_file in history_files:
        loop = int(regex.match(r".*history_([0-9]+).json", history_file).group(1))
        with open(history_file, 'r') as f:
            hist_file = json.loads(f.read())

        trainings.append({
                'info': hist_file['info'],
                'train': hist_file['train'][loop]['history'],
                'eval': hist_file['eval'][loop]
            })

    return trainings

def get_patience(training):
    ''' Figure out what epoch has lowest validation loss
        ASSUMING PATIENCE DOES NOT CHANGE
        early stopping being turned on or off in the same test is fine '''

    if "EarlyStopping" not in training['info']['callbacks']:
        return -1
    
    # Get index of lowest validation loss
    val_loss = training['train']['val_loss']
    return min(enumerate(val_loss), key=lambda i_v: i_v[1])[0] - len(val_loss)

def preprocess_plotting_data(eval_baseline, all_data, metric_keys, info):
    ''' Calculates AUC values, average values and min/max values for all data sets '''

    processed_data  = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    prio_functions = list(all_data.keys())

    for prio_func in prio_functions:
        for data_set in ('train', 'eval'):
            for metric in metric_keys:

                # Calculate x-axis values
                budget_lst = all_data[prio_func]['budget']
                images_seen =  np.fromiter(accumulate(budget_lst, lambda x,y: x+y, initial=0), dtype=int)
                if "n_base" in info and info['n_base'] > 0:
                    images_seen[1:] += info['n_base'] # If n_base, we assume images_seen starts with [0, 0, ...], add baseline count to all but first value
                if (data_set != 'eval') or eval_baseline is None:
                    images_seen = images_seen[1:] # Skip the 0 value, since no baseline data is there

                # Calculate average, min, max and std deviation
                data = all_data[prio_func][data_set][metric]

                nbr_epochs = all_data[prio_func]['nbr_epochs']
                if len(np.array(nbr_epochs).shape) == 2: # list of list
                    epochs = [sum(x)/len(x) for x in zip(*nbr_epochs)]
                else:
                    epochs = nbr_epochs

                # Calculate AUC
                auc = [np.trapz(x, images_seen) for x in data]

                processed_data[prio_func][data_set][metric] = {
                    'avg_data':     [sum(x)/len(x) for x in zip(*data)],
                    'min_data':     [min(x)        for x in zip(*data)],
                    'max_data':     [max(x)        for x in zip(*data)],
                    'std_devi':     [np.std(x)     for x in zip(*data)],
                    'auc'     :     auc,
                    'avg_auc':      np.mean(auc) ,
                    'xs':           images_seen,
                    'epochs':       epochs,
                    'avg_epochs':   sum(epochs)/len(epochs)
                }

    return processed_data

def plot_processed_data(baseline, show_textbox, metric_keys, processed_data, use_standard_deviation, suptitle, info, plot_epochs):
    if plot_epochs:
        width = 200
        offset = get_offset(processed_data)

    for i, prio_func in enumerate(processed_data.keys()):
        for data_set in ('train', 'eval'):
            for metric in metric_keys:
                
                # Get data
                data = processed_data[prio_func][data_set][metric]
                avg_data   = np.array(data['avg_data'])
                min_data   = data['min_data']
                max_data   = data['max_data']
                std_devi   = np.array(data['std_devi'])
                auc        = data['auc']
                xs         = data['xs']
                epochs     = data['epochs']
                avg_epochs = data['avg_epochs']

                # Code to plot epochs
                if prio_func == 'loss_dsc' and data_set == 'eval' and metric == 'loss' and not plot_epochs:
                    plot_epochs_separate(suptitle, info, prio_func, data_set, metric, xs, epochs)

                # Code to plot results
                fig = plt.figure(f'{data_set}_{metric}')
                plot_results(use_standard_deviation, suptitle, info, prio_func, data_set, metric, avg_data, min_data, max_data, std_devi, auc, xs)

                # Plot data
                if plot_epochs:
                    plot_epochs_same_graph(width, offset, i, prio_func, xs, epochs, avg_epochs, fig)

                # Code to plot info box
                if show_textbox:
                    plot_text_box(info, data_set, metric)

def get_offset(processed_data):
    nbr_prio_funcs = len(processed_data.keys())
    start = 0.5 if nbr_prio_funcs % 2 == 0 else 0
    offset = np.array([start+1*i for i in range(math.ceil(nbr_prio_funcs/2))])
    offset = np.concatenate((-1*np.flip(offset), offset))
    if nbr_prio_funcs % 2 != 0:
        offset = np.delete(offset, math.ceil(nbr_prio_funcs/2))
    print(offset)
    return offset

def plot_results(use_standard_deviation, suptitle, info, prio_func, data_set, metric, avg_data, min_data, max_data, std_devi, auc, xs):
    plt.title(f'{data_set}_{metric}')
    plt.xlabel('Images Selected')
    plt.ylabel('Loss' if 'loss' in metric else 'Accuracy')
    if suptitle and suptitle == "ds" and 'datamix' in info: 
        plt.suptitle(f"{info['datamix']} dataset", weight='bold')
    elif suptitle:
        plt.suptitle(suptitle, weight='bold')
    if use_standard_deviation:
        plt.fill_between(xs, avg_data - std_devi, avg_data + std_devi, alpha=.1, linewidth=0)
    else:
        plt.fill_between(xs, min_data, max_data, alpha=.1, linewidth=0)
    plt.plot(xs, avg_data, 'o-', label=f'{prio_func} ({len(auc)})')
    plt.legend(loc=('upper right' if (metric == 'loss') != (data_set == 'train') else 'lower right'))
    plt.grid(True)

def plot_epochs_same_graph(width, offset, i, prio_func, xs, epochs, avg_epochs, fig):
    current = plt.gca()
    if len(fig.axes) == 2:
        plt.sca(fig.axes[1])
    else:
        plt.twinx()
        plt.ylabel('Epochs')
    plt.bar(
                        xs + np.array(width * offset[i]),
                        epochs,
                        width=width,
                        alpha=0.2,
                        label=f'{prio_func}_avg: {str(round(avg_epochs))}',
                    )
    plt.legend(loc=('upper left'))
    plt.sca(current)

def plot_text_box(info, data_set, metric):
    text_str = '\n'.join((
                        r'$\mathrm{total}=%i$'       % info['train_count'],                 # total can be seen in x-axis
                        r'$\mathrm{budget}=%i$'      % info['sample_count'],                # budget can be seen in x-axis
                        # r'$\mathrm{epochs}=%i$'      % info['epochs'],                    # not checked if same for all
                        # r'$\mathrm{shuffled}=%s$'    % info['shuffled'],                  # always true
                        # r'$\mathrm{fixed-steps}=%s$' % info['fixed_steps_per_epoch'],     # always true
                    ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.2)
    point = (0.025, 0.125) if (metric == 'loss') != (data_set == 'train') else (0.025, 0.975)
    text  = plt.text(
                        point[0],
                        point[1],
                        text_str,
                        fontsize=8,
                        verticalalignment='top',
                        transform=plt.gca().transAxes,
                        bbox=props)
    text.set_bbox(dict(facecolor='white', alpha=0.2, linewidth=1))

def plot_epochs_separate(suptitle, info, prio_func, data_set, metric, xs, epochs):
    plt.figure('epochs')
    width = 250
    plt.bar(xs[1:], epochs, width=width, alpha=1, label=prio_func)
    plt.legend(loc=('upper left'))
    plt.xlabel('Images Selected')
    plt.ylabel('Avg. Epochs')
    plt.title(f'{data_set}_{metric}')
    if suptitle and suptitle == "ds" and 'datamix' in info: 
        plt.suptitle(f"{info['datamix']} dataset", weight='bold')
    elif suptitle:
        plt.suptitle(suptitle, weight='bold')

def plot_OD_metrics(history_rel_paths, prio_fns, plot_metrics=None, eval_baseline=None, use_standard_deviation=False):
    ''' Plots files containing OD metrics in the given list of paths. Produces a plot for mAP50 and AP50.
        If multiple files have the same selection method, they are plotted together as average + std deviation range.
        Supports including values from initial evaluation of baseline.
    '''

    '''
    Example: 
    {
        random: {
            mAP50: {
                [[1,2,3], [4,5,6]]
            },
            AP: {
                bus: [[1,2,3], [4,5,6]],
                human: [[1,2,3], [4,5,6]]
            },
            nbr_loops: int
        },
    }
    '''
    assert len(history_rel_paths[0]) != 0, 'Not enough files provided, list of lists needed'
    assert len(history_rel_paths) == len(prio_fns), 'Nbr of files and nbr of prio_fns does not match'

    nbr_files = len(history_rel_paths[0]) + 1

    all_data = defaultdict(
        lambda: {
            'mAP50': [],
            'AP50': defaultdict(list),
            'nbr_loops': nbr_files,
        }
    )

    metric_keys = []
    # Read all files and save the data
    for history_rel_path, prio_func in zip(history_rel_paths, prio_fns):

        mAP_results = []
        AP_results = defaultdict(list)
        for path in history_rel_path:

            with open(path, 'r') as f:
                obj_str = f.read()
                file = json.loads(obj_str)

            # Get keys for metrics
            mAP_res = file['default']['default_eval_none']['default']['mAP50'] # value
            mAP_results.append(mAP_res)

            metric_keys = file['default']['default_eval_none']['default']['AP50'].keys()
            for metric in metric_keys:
                AP_results[metric].append(file['default']['default_eval_none']['default']['AP50'][metric])

        all_data[prio_func]['mAP50'].append(mAP_results)
        for metric in metric_keys:
            all_data[prio_func]['AP50'][metric].append(AP_results[metric])

    if eval_baseline is not None:
        with open(eval_baseline, 'r') as f:
            obj_str = f.read()
            file = json.loads(obj_str)

        base_mAP_res    = file['default']['default_eval_none']['default']['mAP50']
        base_AP_results = {
            metric: file['default']['default_eval_none']['default']['AP50'][
                metric
            ]
            for metric in metric_keys
        }
    # Load the saved data and plot it
    if plot_metrics is not None:
        metric_keys = plot_metrics

    sorted_keys = list(all_data.keys())
    sorted_keys.sort(key=lambda key: Prio[key.split('-')[0]].value)

    for prio_func in sorted_keys:

        nbr_loops = all_data[prio_func]['nbr_loops']
        # Plot mAP50
        data = all_data[prio_func]['mAP50']
        start = 1
        start, avg_data = prep_and_plot_ranges(eval_baseline, use_standard_deviation, nbr_loops, data, base_mAP_res)

        conf_figure_args('mAP50', 'mAP50')
        plt.plot(
            range(start, nbr_loops),
            avg_data,
            'o-',
            label=f'{prio_func} ({len(data)})',
        )

        # Plot AP50
        for metric in metric_keys:
            data = all_data[prio_func]['AP50'][metric]
            results = base_AP_results[metric]
            start, avg_data = prep_and_plot_ranges(eval_baseline, use_standard_deviation, nbr_loops, data, results)
            
            conf_figure_args('AP50', 'AP')
            plt.plot(
                range(start, nbr_loops),
                avg_data,
                'o-',
                label=f'{prio_func}_{metric} ({len(data)})',
            )


    plt.show()

def conf_figure_args(title, y_label):
    plt.figure(title)
    plt.title(title)
    plt.xlabel('Loop Nbr')
    plt.ylabel(y_label)
    plt.legend(loc='lower right')
    plt.grid(True)

def prep_and_plot_ranges(eval_baseline, use_standard_deviation, nbr_loops, data, results):
    avg_data = [sum(x)/len(x) for x in zip(*data)]
    min_data = [min(x) for x in zip(*data)]
    max_data = [max(x) for x in zip(*data)]
    std_devi = [np.std(x)     for x in zip(*data)]

    if eval_baseline is not None:
        avg_data.insert(0, results)
        min_data.insert(0, results)
        max_data.insert(0, results)
        std_devi.insert(0, results)
        start = 0

    if use_standard_deviation:
        avg_data = np.array(avg_data)
        std_devi = np.array(std_devi)
        plt.fill_between(range(start, nbr_loops), avg_data - std_devi, avg_data + std_devi, alpha=.1, linewidth=0)
    else:
        plt.fill_between(range(start, nbr_loops), min_data, max_data, alpha=.1, linewidth=0)
    return start,avg_data