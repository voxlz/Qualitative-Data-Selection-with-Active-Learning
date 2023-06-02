

from collections import defaultdict
import math
import tensorflow as tf
from models.oracle import Oracle
from utils.tf_utils import imagenet_decode_predictions


# Confidence Selection Methods
def least_conf(img_score, n_labels):
    ''' Calculate the difference between 100% and highest confidence score.'''
    return (1 - img_score) * (n_labels/(n_labels-1))

def margin_conf(top_score, sec_score):
    ''' Calculate the difference between the top two predicted scores. '''
    return 1 - (top_score - sec_score)

def entropy(prob_dist):
    ''' Calculate the difference between the top two predicted scores. '''
    log_probs = [math.log2(x)*x if x > 0.0 else 0 for x in prob_dist]
    return (0 - sum(log_probs)) / math.log2(len(prob_dist))



# Loss Selection Methods
def sel_with_oracle(model, img_loss_ds, unseen_data, n_sample, n_batch, n_epochs=10):
    ''' Select n_sample images from unseen_data using oracle model. '''

    oracle = Oracle(model)
    print("Training Oracle Model to predict loss")
    oracle.fit(img_loss_ds, epochs=n_epochs, batch_size=n_batch)
    y_pred = oracle.predict(unseen_data.batch(n_batch))
    selected = tf.argsort(y_pred, axis=0, direction='DESCENDING').numpy()[:n_sample]
    return selected



# Help Functions
def margin_conf_uncert(y_pred, _):
    ''' y_pred expects list of top scores, only requires top 2 '''
    # assert len(y_pred[0]) >= 2, "Need at least 2 scores per prediction"
    return [margin_conf(scores[0], scores[1]) for scores in y_pred]

def least_conf_uncert(y_pred, n_labels):
    ''' y_pred expects list of top scores, only requires top 1 '''
    # assert len(y_pred[0]) >= 1, "Need at least 1 score per prediction"
    return [least_conf(img_scores[0], n_labels) for img_scores in y_pred]

def entropy_uncert(y_pred, _):
    ''' y_pred expects list of top scores, requires all scores '''
    return [entropy(prob) for prob in y_pred]

def diversity_sampling(prio_order, unseen_data, y_pred, leniency=1):
    ''' While respecting the original order, prioritize images that are predicted to be from different classes.
    
    Args:
        - prio_order:     List of indices that represent the original order of the images
        - unseen_data:    Dataset of images that are not yet labeled
        - y_pred:         List of predictions for each image in unseen_data
        - leniency:       How many images per class should be selected. Negative values will select based on true labels.
    
    Returns: reordered prio_order
    '''
    
    if leniency == 0: return prio_order
    
    max_lbl_count   = abs(leniency)

    # Keep track what has been selected
    selected_class_count = defaultdict(int)
    selected_idxs = []

    # Calculate top class prediction for each image
    if (leniency < 0):
        top_pred_per_image = [x[1] for x in unseen_data.as_numpy_iterator()]
        leniency *= -1
    else:
        top_pred_per_image = [pred_per_image[0][3] for pred_per_image in y_pred]

    not_selected_idxs = prio_order
    while len(not_selected_idxs) > 0:
        new_not_selected_idxs = []
        for image_idx in not_selected_idxs:
            label = top_pred_per_image[image_idx]
            if selected_class_count[label] < max_lbl_count:
                selected_idxs.append(image_idx)
                selected_class_count[label] += 1
            else:
                new_not_selected_idxs.append(image_idx)
        not_selected_idxs = new_not_selected_idxs
        max_lbl_count += 1
    
    return selected_idxs

