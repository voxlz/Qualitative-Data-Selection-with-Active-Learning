from enum import IntEnum, auto
import json
import os
import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from collections import defaultdict
from active_learning.active_learning_methods import diversity_sampling, entropy_uncert, least_conf_uncert, margin_conf_uncert
from unused.ssd_code.eval import get_predictions
from PIL import Image
from utils.ssd_utils import batch_dataset, dataset_to_ground_truth, get_coco_raw, get_subset_dataset, get_subsets_dataset, preprocess_coco
from utils.tf_utils import imagenet_decode_predictions

from vgg16.vgg16_utils import deprocess_np_img

Concatenate = tf.keras.layers.concatenate
Dataset = tf.data.Dataset

class Prio(IntEnum):
    least_conf  = auto()
    margin_conf = auto()
    entropy     = auto()
    loss_dsc    = auto()
    random      = auto()
    loss_asc    = auto() # Deprecated

def get_sample_oracle(oracle, dataset, n_sample, default_boxes):
    images, _, _, _ = dataset_to_ground_truth(dataset, default_boxes)
    y = oracle(images)

    idxs_max_loss = tf.argsort(y, axis=0, direction='DESCENDING').numpy()[:n_sample]
    return get_subset_dataset(dataset, idxs_max_loss)

def get_sample_confidence(model, dataset, n_sample, batch):
    batches = batch_dataset(dataset, batch=batch)
    uncertainty = []

    for batch in tqdm(batches, desc="Extracting images", position=0):
        images, bboxes, labels = get_coco_raw(batch)
        images, _, labels, _ = preprocess_coco(images, bboxes, labels)
        # Perform Inference
        scores, offsets = model(images)
        scores  = tf.math.softmax(scores, axis=2)
        # TODO - Tune softmax?

        # Find most prominent class score for every prediction
        _, default_boxes = model.get_default_boxes()
        labels, _, scores = get_predictions(scores, offsets, default_boxes)
        avg_scores = [tf.math.reduce_mean(img_scores, axis=0) for img_scores in scores]

        # Least confidence
        n_labels = labels[0].shape[-1]
        img_uncertainty = [(1 - img_score)*(n_labels/(n_labels-1)) for img_score in avg_scores]

        uncertainty.append(img_uncertainty)

    uncertainty = [item for batch_uncertainty in uncertainty for item in batch_uncertainty] #flatten the list

    img_sort_order = tf.argsort(uncertainty, direction='DESCENDING').numpy()[:n_sample]

    data_sample = get_subset_dataset(dataset, img_sort_order)
    return data_sample, img_sort_order, uncertainty

def restore_loss_dict(loss_funcs): 
    temp_dict = defaultdict(lambda: None)
    for loss_fn in loss_funcs:
        if loss_fn is None:
            temp_dict['anchors'] = None
            continue
        name = loss_fn.get_config()['name']

        if 'bounding_box' in name:
            temp_dict['box_encodings'] = loss_fn
        elif 'focal_loss' in name:
            temp_dict['class_predictions'] = loss_fn
        elif 'categorical_attribute' in name:
            if temp_dict['vehicle_color_predictions'] is None:
                temp_dict['vehicle_color_predictions'] = loss_fn
            elif temp_dict['clothing_upper_color_predictions'] is None:
                temp_dict['clothing_upper_color_predictions'] = loss_fn
            elif temp_dict['clothing_lower_color_predictions'] is None:
                temp_dict['clothing_lower_color_predictions'] = loss_fn
    return dict(temp_dict) 

# @profile
def select_samples_with_prio(
    prio: str, 
    unseen_data, 
    budget, 
    n_batch, 
    model, 
    model_post_proc=None, 
    save_dir="/", 
    AL_loop=-1, 
    sel_pivot: float = 0,
    save_imgs: bool = False,
    save_top_x_imgs: int = 0,
    n_unseen: int = None,
    diversity: int = 0,
):
    ''' Select n_sample images from unseen_data using the given prioritization function.

    Args:
    - prio: string indicating which prioritization function to use
    - unssen_data: data pool to make the seleciton from
    - budget: refers to how many data to include in the selection
    - n_batch: batch size
    - model: current model used
    - model_post_proc: if using an OD model, then this is the same model with post processing applied
    - save_dir: save location
    - AL_loop: the current outer loop number
    - sel_pivot: float value between 0-1, works as an anchorpoint for the selection. 0 pick the hardest and 1 the easiest
    - save_imgs: if selected images should be saved
    - save_top_x_imgs: how many top images to save, if 0 none will be saved
    - n_unseen: how many data in the unseen_data pool
    - diversity: if 0 no diversity selction, otherwise will diversify the selction with this number per class 

    Returns:
    - data_samples: tf.data.Dataset with selected images
    - unseen_data: tf.data.Dataset with remaining images'''

    def get_od_loss(x, y): # used with OD model, makes use of the model's compiled loss functions
        ''' Given a batched dataset with (x, y) pairs, return batched loss'''
        y_pred = model(x)
        loss_dict = model.compiled_loss._losses
        if type(loss_dict) == list:
            loss_dict = restore_loss_dict(loss_dict)               
        losses = [[loss_dict[key](tf.expand_dims(y[key][img], axis=0), tf.expand_dims(y_pred[key][img], axis=0))
                  for key in y.keys()] for img in range(n_batch)]
        loss = tf.reduce_sum(losses, axis=-1)
        loss = tf.keras.activations.sigmoid(loss/17-4)
        return loss

    # Get predictions
    selection = get_selection(prio, budget, unseen_data, model, n_batch, n_unseen, sel_pivot, diversity, model_post_proc, get_od_loss)

    if save_imgs:
        save_sample_imgs(unseen_data, save_dir, AL_loop, selection)
    
    # Save images
    if save_top_x_imgs:
        save_top_imgs(unseen_data, save_dir, AL_loop, selection, save_top_x_imgs)
    
    # Return selected images + remaining images
    return get_subsets_dataset(unseen_data, selection)

def get_selection(prio, budget, unseen_data, model, n_batch, n_unseen, sel_pivot=0, diversity=0, model_post_proc=None, get_od_loss=None):
    if model_post_proc: # Object Detection
        y_pred      = model_post_proc.predict(unseen_data.batch(n_batch))
        n_labels    = 8
    else:
        y_pred      = model.predict(unseen_data.batch(n_batch))
        n_labels    = y_pred.shape[1]

    prio_order = uncertainty_sampling(prio, unseen_data, n_batch, n_unseen, n_labels, y_pred, model_post_proc, get_od_loss)

    prio_order = pivot_reorder(prio_order, budget, sel_pivot)
    
    if model_post_proc is None:
        prio_order = diversity_sampling(prio_order, unseen_data, imagenet_decode_predictions(y_pred, top=1), leniency=diversity)
    
    return prio_order[:budget]

def prune_samples(prio, n_prune, unseen_data, model, n_batch, n_unseen):
    prune_selection = get_selection(prio, n_prune, unseen_data, model, n_batch, n_unseen)
    return get_subsets_dataset(unseen_data, prune_selection)

def pivot_reorder(prio_order, size, pivot):
    ''' Create a slice around the pivot (in range [0,1]) with specified size. Move this slice to the front of the array. 
    
        NOTE: Even sizes will be rounded up to the nearest odd number.
    '''
    assert 0 <= pivot <= 1, "Pivot value must be between 0 and 1"
    assert size <= len(prio_order), "Size must be smaller than or equal to the length of the array"
    
    pivot = int(pivot * len(prio_order))
    half_budget = size // 2
    
    if (pivot + half_budget > len(prio_order)-1):
        pivot -= ((pivot + half_budget) - (len(prio_order)-1))
    if (pivot - half_budget < 0):
        pivot += (half_budget - pivot)
    
    prio_start = pivot - half_budget
    prio_end   = pivot + half_budget
    
    return np.append(prio_order[prio_start:prio_end+1], np.append(prio_order[:prio_start], prio_order[prio_end+1:]))

def save_sample_imgs(unseen_data, save_dir, AL_loop, selection):
    
    save_img(save_dir, AL_loop, unseen_data, 'hard', selection[:1])
    save_img(save_dir, AL_loop, unseen_data, 'easy', selection[-1:])
    
def save_top_imgs(unseen_data, save_dir, AL_loop, selection, n_imgs=100):
    # assert n_imgs <= len(selection), "n_imgs must be smaller than or equal to the length of the array"
    
    save_img(save_dir, AL_loop, unseen_data, 'top_selected', selection[:n_imgs])

def save_img(save_dir, AL_loop, unseen_data, name, order):
    data, _     = get_subsets_dataset(unseen_data, order)
    
    for i, (x, y) in enumerate(data):
        img = deprocess_np_img(x.numpy())
        img = Image.fromarray(img)
                        
        label = y.numpy()
        with open("dataset_labels/imagenet_class_index.json") as f:
            CLASS_INDEX = json.load(f)
            
        os.makedirs(save_dir, exist_ok=True)
        class_label = CLASS_INDEX[str(label)][1] if CLASS_INDEX is not None else ""
        img.save(os.path.join(save_dir, "%d_%s_%d_%s.png" % (AL_loop, name, i, class_label)))

def uncertainty_sampling(prio, unseen_data, n_batch, n_unseen, n_labels, y_pred, model_post_proc=None, get_od_loss=None):
    p = prio
    
    if p == Prio.random:
        if model_post_proc is not None:
            prio_order = random.sample(range(n_unseen), n_unseen)
        else:
            prio_order = tf.random_index_shuffle(range(n_unseen), [10,12,15], n_unseen-1).numpy()
    elif p == Prio.least_conf:
        prio_order = select_with(least_conf_uncert, y_pred, n_labels, model_post_proc)
    elif p == Prio.margin_conf:
        prio_order = select_with(margin_conf_uncert, y_pred, n_labels, model_post_proc)
    elif p == Prio.entropy:
        prio_order = select_with(entropy_uncert, y_pred, n_labels, model_post_proc)
    elif p == Prio.loss_dsc:
        if model_post_proc is not None: # Object Detection
            losses     = (get_od_loss(x, y) for x, y in tqdm(unseen_data.batch(n_batch), desc="Converting losses"))
            losses     = [loss for sublist in losses for loss in sublist]
        else: # Classification
            # Batch predictions
            pred_tens = tf.convert_to_tensor(y_pred)
            pred_tens = tf.reshape(y_pred, (int(y_pred.shape[0] / n_batch), n_batch, n_labels))
            
            loss_fn = tf.losses.SparseCategoricalCrossentropy(reduction='none')
            @tf.function
            def get_classification_loss_v2(i, xy):
                ''' Given a batched dataset with (x, y) pairs, return batched loss'''
                return loss_fn(xy[1], pred_tens[i])
            
            losses = unseen_data.batch(n_batch).enumerate().map(
                get_classification_loss_v2, 
                num_parallel_calls=16
            ).unbatch()

            losses  = tqdm(losses, desc="Extracting losses", total=n_unseen)
            losses  = np.fromiter(losses, dtype=np.float32)

        prio_order = np.argsort(losses, axis=0).astype(np.int32)
        prio_order = np.flip(prio_order, axis=0) # Descending order
        del losses
    else:
        raise NotImplementedError(f"Prioritization function {p} not implemented.")
    
    assert n_unseen == len(prio_order), f"Length of unseen data ({n_unseen}) and prio_order ({len(prio_order)}) does not match."
    return prio_order

def select_with(pred_to_uncert_fn, y_pred, n_labels: int = 1000, model_post_proc=None):
    ''' Select n_sample images from unseen_data using the given uncertainty function.

        Diversity describes the leniency in the diversity sampling. A value of 0 means no diversity sampling. A value of x < 0 means perfect diversity sampling.'''
    if model_post_proc is not None: # Object Detection
        uncertainty = []

        for idx in range(len(y_pred['num_boxes'])):
            nbr_detections = y_pred['num_boxes'][idx]
            if nbr_detections == 0:
                uncertainty.append(1) # high uncertainty, didn't find any detections
                continue

            img_confidences = y_pred['top_k_confidences'][idx][:nbr_detections]
            img_confidences = [conf.tolist() for conf in img_confidences]
            final_score = pred_to_uncert_fn(img_confidences, n_labels)

            uncertainty.append(sum(final_score) / len(final_score)) # avg over boxes
    else: # Classification
        y_pred      = imagenet_decode_predictions(y_pred, top=-1)
        y_pred_scores = [[tuple[2] for tuple in conf_list] for conf_list in y_pred] # classification
        uncertainty = pred_to_uncert_fn(y_pred_scores, n_labels)

    prio_order = tf.argsort(uncertainty, axis=0, direction='DESCENDING').numpy()
    
    assert uncertainty[prio_order[0]] == max(uncertainty)

    return prio_order
