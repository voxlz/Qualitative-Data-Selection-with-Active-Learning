
# Get model and dataset. Create data prioritization function (oracle, confidence, etc.).  Extract data from "unlabeled" dataset.
# Perform training with new data. Save results. LOOP
import functools
import gc
import os
import time
import tensorflow as tf
from tqdm import tqdm
import json
from typing import List
from active_learning.active_learning_utils import prune_samples, select_samples_with_prio
from vgg16.vgg16_datamix import Datamix

keras = tf.keras
Callback = tf.keras.callbacks.Callback
Model = tf.keras.Model
Dataset = tf.data.Dataset

def active_learning(
    model           : Model,
    model_post_proc : Model,
    vali_data       : Dataset,
    unseen_data     : Dataset,
    test_data       : Dataset,
    base_data       : Dataset           = None,
    n_unseen        : int               = None,
    n_loops         : int               = None,
    save_dir        : str               = None,
    log_dir         : str               = None,
    weight_dir      : str               = None,
    extra_callbacks : List[Callback]    = None,
    eval_dir        : str               = None,             # Used with an OD model
    eval_fn         : functools.partial = None,             # Used with an OD model
    sel_pivot       : float             = 0,
    budget          : int or List[int]  = 500,
    n_epoch         : int               = 5,
    n_batch         : int               = 20,
    prio            : str               = "random",
    fixed_steps     : bool              = False,            # Use fixed steps per epoch
    diversity       : int               = 0,                # Enable diversity sampling
    datamix         : Datamix           = Datamix.undefined,
    n_prune         : int               = 0,                # Remove hardest images from training set after training on them
    accumulate      : bool              = True,             # Accumulate data into seen_data from previous training steps
    n_base          : int               = 0,                # Number of base samples to use
):
    ''' Active learning loop. Used to simulate active learning with human labeling or virtual data synthesis.
        Make sure model expects the input shape of dataset.'''

    # Make sure directories are specified
    assert save_dir    is not None, "save_dir   must be specified."
    assert log_dir     is not None, "log_dir    must be specified."
    assert weight_dir  is not None, "weight_dir must be specified."

    start_time                      = time.time()
    total_train_count               = n_unseen
    eval_lst_of_dicts               = []
    train_history                   = []
    budget_acc                      = [sum(budget[:i+1]) for i in range(len(budget))]
    seen_data   : tf.data.Dataset   = base_data # None if base_data is None
    pruned_data : tf.data.Dataset   = None

    if n_loops  is None: n_loops  = len(budget)

    tf.keras.backend.clear_session()
    model.load_weights(weight_dir)

    # Evaluate and save results
    # eval_result = model.evaluate(val_data.batch(n_batch), return_dict=True)
    # eval_lst_of_dicts.append(eval_result)

    # Active learning loop
    for i in tqdm(range(n_loops), position=0, desc=f"{prio} learning loop", leave=True, colour="#25cfe1"):
        print("Active learning loop %d" % i)

        # Select new samples with previous loops trained model
        if accumulate:
            print("select data accumulatively")
            data_samples, unseen_data = select_samples_with_prio(
                prio, 
                unseen_data, 
                budget[i], 
                n_batch, 
                model, 
                model_post_proc, 
                save_dir, 
                AL_loop=i,
                sel_pivot=sel_pivot,
                n_unseen=n_unseen,
                diversity=diversity,
                save_top_x_imgs=5,
            )
            n_unseen -= budget[i]
            seen_data = data_samples if seen_data is None else seen_data.concatenate(data_samples)
            del data_samples
        else:
            selected_ds, _ = select_samples_with_prio(
                prio, 
                unseen_data, 
                budget_acc[i], 
                n_batch, 
                model, 
                model_post_proc, 
                save_dir, 
                AL_loop=i,
                sel_pivot=sel_pivot,
                n_unseen=n_unseen,
                diversity=diversity,
            )
            seen_data = base_data.concatenate(selected_ds)

        # Reset weights on model
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        gc.collect()
        print("load weights")
        model.load_weights(weight_dir)

        # Tensorboard callback
        log_path = os.path.join(log_dir, "AL_loop_%d" % i)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1, profile_batch = '0,10')
        if extra_callbacks is not None:
            callbacks = extra_callbacks + [tensorboard_callback]
            print("callbacks: ", extra_callbacks)
        else:
            callbacks = [tensorboard_callback]

        # Train and evaluate
        half_loops = n_loops // 2
        step_per_epoch = int(budget[i]*half_loops/n_batch)
        buff = sum(budget[:i+1])
        train_data = seen_data.shuffle(buffer_size=(buff+n_base))
        train_data = train_data.repeat() if fixed_steps else train_data
        train_result = model.fit(
            train_data.batch(n_batch).prefetch(4),
            epochs          = n_epoch,
            steps_per_epoch = step_per_epoch if fixed_steps else None,
            callbacks       = callbacks,
            validation_data = vali_data.batch(n_batch),
            shuffle         = False,
        )
        eval_result = model.evaluate(test_data.batch(n_batch), return_dict=True, callbacks=callbacks)

        # Prune hardest samples (optional)
        if not model_post_proc and n_prune > 0:
            print("pruning data")
            pruned, unseen_data = prune_samples(prio, n_prune, unseen_data, model, n_batch, n_unseen)
            pruned_data = pruned if pruned_data is None else pruned_data.concatenate(pruned)

        # Save history and training data
        eval_lst_of_dicts.append({k: float(v) for k, v in eval_result.items()})
        train_history.append({
            "history": {k: [float(val) for val in v] for k, v in train_result.history.items()}, 
            "params": train_result.params
        })

        loop_str = "loop %d" % (i)
        if model_post_proc is not None: # Object Detection, save OD metrics
            tf.keras.backend.clear_session() 
            print("Evaluating...")
            eval_fn((eval_dir/ loop_str))
            print("Done evaluating.")

        # Collect data
        train_info = {
            'info': {
                "prio_func": prio.name,
                "train_count": total_train_count,
                "sample_count": budget[i],
                "epochs": n_epoch,
                "time_in_min": "{:.2f}".format(
                    (time.time() - start_time) / 60
                ),
                "shuffled": True,
                "callbacks": [c.__class__.__name__ for c in callbacks],
                "model_name": "axisOD" if model_post_proc else "vgg16",
                "pivot": sel_pivot,
                "fixed_steps": fixed_steps,
                "diversity": diversity,
                "datamix": datamix,
                "accumulate": accumulate,
                "n_base": n_base,
            },
            'train': train_history,
            'eval': eval_lst_of_dicts,
        }
        # Save to file
        os.makedirs(os.path.join(save_dir), exist_ok=True)
        with open(os.path.join(save_dir, "history_%d.json" % i), "w") as f:
            f.write(json.dumps(train_info))

        gc.collect()
        tf.keras.backend.clear_session()

    return seen_data
