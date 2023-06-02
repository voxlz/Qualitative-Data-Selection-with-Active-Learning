import os
import random
import time

import numpy as np
import tensorflow as tf
from functools import partial
from tqdm import tqdm

from utils.ssd_utils import batch_dataset, dataset_to_ground_truth

def train(model, optimizer, train_ds, weights_path, n_imgs=10, batch_size=10, n_epoch=50, inter_save=5, input_shape=(300,300,3)):
    """
    Method to train an SSD architecture
    
    Args:
        - model: Keras SSD model
        - optimizer: optimizer to use
        - train_ds: dataset to train on
        - weights_path: path to save weights
        - n_imgs: number of images to train on
        - batch_size: batch size
        - n_epoch: number of epochs
        - inter_save: interval to save weights
        - input_shape: input shape of images
    """
    
    assert batch_size <= n_imgs
    _tqdm = partial(tqdm, position=0, leave=True)
    _, default_boxes = model.get_default_boxes()

    # Shuffle, take, and batch dataset
    train_ds = train_ds.take(n_imgs)
    train_ds = [x for x in train_ds]  # load in dataset to memory, so it can be shuffled
    
    print("Started training!")
    
    for epoch in _tqdm(range(n_epoch)):
        losses = []

        # batches the data. Will be shuffled to make SGD. VALIDATED
        random.shuffle(train_ds)
        ds = batch_dataset(train_ds, batch_size)

        for batch in tqdm(ds):
            print(len(batch))

            # get data from batch
            a1 = time.time()
            imgs, scores, offsets, _ = dataset_to_ground_truth(batch, default_boxes, input_shape)
            print("Time to get ground truth: {}".format(time.time() - a1))

            # get predictions and losses
            a2 = time.time()
            with tf.GradientTape() as tape:
                
                # Forward pass
                p_scores, p_offsets = model(imgs)

                # calculate loss
                model_loss = model.get_loss(p_scores, scores, p_offsets, offsets)

                # L2 regularization ???
                l2 = [tf.nn.l2_loss(t).numpy() for t in model.trainable_variables]
                model_loss = model_loss + 0.001 * tf.math.reduce_sum(l2)
                
                # if (model_loss > 1000):
                #     print("Loss is too high: {}".format(model_loss))
                losses.append(model_loss)
            print("Time to record tape: {}".format(time.time() - a2))
    
            # back propagation (outside with-block according to convention) 
            a3 = time.time()
            grads = tape.gradient(model_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            print("Time to calc and apply gradients: {}".format(time.time() - a3))

        print("Mean loss: {} on epoch {}".format(np.mean(losses), epoch))
        if epoch % inter_save == 0 and epoch > 0:
            path = os.path.join(weights_path, "ssd_weights_epoch_{:03d}.h5".format(epoch))
            print(path)
            model.save_weights(path)
            
    print("Done training!")
