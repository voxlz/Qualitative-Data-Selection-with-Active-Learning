import json
import os
from datetime import datetime
import tensorflow as tf
from active_learning.active_learning import active_learning
from active_learning.active_learning_utils import Prio
from utils.image_net_utils import prep_budget
from models.vgg16 import VGG16
from tqdm import tqdm
from vgg16.vgg16_datamix import Datamix
from vgg16.vgg16_dataset_utils import get_dataset

''' Main script for running active learning experiments on the VGG16 image detection model. '''

Dataset = tf.data.Dataset

# GPU Debug 
cpu_only = False
if cpu_only:
    tf.config.set_visible_devices([], 'GPU') # DISABLE GPU
    for _ in range(100):
        print('CPU ONLY BITCH')
else:
    gpus = tf.config.list_physical_devices('GPU')
    print(gpus)
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)
    del gpus

# Seed for reproducibility
tf.random.set_seed(2541)

# Suppress warnings like 10% of memory left
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main(save_baseline=True):
    ''' Main function. Defines the test and starts the active learning loops. '''
    
    # Define callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, min_delta=0.0001, restore_best_weights=True, verbose=1, mode='min')
    # early_stop_acc  = tf.keras.callbacks.EarlyStopping(monitor='val_top_5_accuracy', patience=2, min_delta=0.0001, restore_best_weights=True, verbose=1, mode='max')

    # Test configuration
    test_name  = 'mixed_1000+5000'
    n_run_test = 3

    # Configure active learning properties
    methods = [
        Prio.least_conf,
        Prio.margin_conf,
        Prio.loss_dsc,
        Prio.entropy,
        Prio.random,
    ]

    conf = {
        "n_unseen"    : 9998,   # ~80% of the dataset
        "n_vali"      : 1,   # ~10% of the dataset 
        "n_test"      : 1,   # ~10% of the dataset
        "n_base"      : 0,    # base training set
        "use_weights" : True,   # Use imagenet weights
        "n_batch"     : 5, 
        "datamix"     : Datamix.mixed,
        "methods"     : methods                            * n_run_test,  
        "budget"      : [[0] + [500]*10]    * len(methods) * n_run_test,
        "n_epoch"     : [1000]              * len(methods) * n_run_test,  # use 1000 epochs for early stopping
        "callbacks"   : [[early_stop]]      * len(methods) * n_run_test,
        "rnd_budget"  : [False]             * len(methods) * n_run_test,
        "sel_pivot"   : [0]                 * len(methods) * n_run_test,  # 0 = hardest data, 0.x = somewhat hard, 1 = easiest data
        "fixed_step"  : [False]             * len(methods) * n_run_test,
        "diversity"   : [0]                 * len(methods) * n_run_test,  # 0 = no diversity, 1 = diversity, -1 = diversity with gt
        "accumulate"  : [True]              * len(methods) * n_run_test,  # Accumulate images or reselect from unseen every loop 
        "n_prune"     : [0]                 * len(methods) * n_run_test,  # 0 = no pruning, >0 = prune n_prune hardest examples every loop
    }

    # Pre-trained VGG16 model with imagenet weights
    print("Loading VGG16 model...")
    weights = 'imagenet' if conf['use_weights'] else None
    vgg16 = VGG16(
        include_top=True,
        weights=weights,
        input_shape=(224, 224, 3),
        classifier_activation='softmax',
    )

    # Choice of dataset
    n_total = conf['n_unseen'] + conf['n_test'] + conf['n_vali'] + conf['n_base']
    dataset, n_images = get_dataset(conf)
    # Prepare budget schedule
    if type(conf['budget']) is int: conf['budget'] = [conf['budget']] * len(conf['methods'])
    if type(conf['rnd_budget']) is bool: conf['rnd_budget'] = [conf['rnd_budget']] * len(conf['methods'])
    budget_schedule = [
        prep_budget(
            budget,
            conf['n_unseen'],
            conf['rnd_budget'][i],
            conf['n_batch'],
            conf['n_epoch'][i],
        )
        for i, budget in enumerate(conf['budget'])
    ]
    # Sanity checks
    assert n_total <= n_images, 'Total must be less than n_images'
    assert len(conf['methods']) == len(budget_schedule) == len(conf['n_epoch']) == len(conf['diversity']) == len(conf['rnd_budget']) == len(conf['callbacks']) == len(conf['fixed_step']) == len(conf['sel_pivot']) == len(conf['n_prune']) == len(conf['accumulate']), 'Configs must be the same length'

    # Generate paths for saving and logging
    dt_str = datetime.now().strftime(f'%Y-%m-%d_%H-%M-%S__{test_name}')

    # Calculate validation baseline
    print("Saving Baseline Evaluation...")
    if save_baseline:
        eval_base_dir   = os.path.join('saved', dt_str, 'eval_base')
        path            = os.path.join(eval_base_dir, 'vgg16_baseline.json')
        baseline_eval   = vgg16.evaluate(dataset['test'].batch(batch_size=int(conf['n_batch'])), verbose=1, return_dict=True)

        # Save baseline
        os.makedirs(eval_base_dir, exist_ok=True)
        with open(path, 'w') as f: 
            f.write(json.dumps(baseline_eval))

    # Run active learning
    print("Running Active Learning...")
    for i, prio_func in tqdm(enumerate(conf['methods']), desc=test_name, total=len(conf['methods']), colour='#e125d1', position=0):
        # Printing something is required for tqdm to not overwrite the previous line
        print(f'new method: {prio_func.name}')

        # Generate paths for saving and logging
        save_dir    = os.path.join('saved', dt_str, '%d. %s' % (i, prio_func.name))
        log_dir     = os.path.join('logs' , dt_str, '%d. %s' % (i, prio_func.name))
        weight_dir  = os.path.join('saved_models', 'vgg16', 'baseline_weights.h5')

        active_learning(
            model           = vgg16,
            prio            = prio_func, 
            unseen_data     = dataset['unseen'],
            vali_data       = dataset['vali'], 
            test_data       = dataset['test'], 
            base_data       = dataset['base'],
            n_unseen        = conf['n_unseen'],
            n_epoch         = conf['n_epoch'][i], 
            n_batch         = conf['n_batch'], 
            n_prune         = conf['n_prune'][i],
            n_base          = conf['n_base'],
            sel_pivot       = conf['sel_pivot'][i],
            fixed_steps     = conf['fixed_step'][i],
            extra_callbacks = conf['callbacks'][i] if dataset['vali'] else None,
            diversity       = conf['diversity'][i],
            accumulate      = conf['accumulate'][i],
            datamix         = conf['datamix'],
            budget          = budget_schedule[i], 
            save_dir        = save_dir, 
            log_dir         = log_dir,
            weight_dir      = weight_dir,
            n_loops         = None, 
            model_post_proc = None, 
        )
        
if __name__ == "__main__":
    main()