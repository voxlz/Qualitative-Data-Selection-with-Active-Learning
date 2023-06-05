import os
from datetime import datetime

import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
from tqdm import tqdm

from active_learning.active_learning_utils import *
from ssd_code.eval import visualize_ground_truth
from models.oracle import Oracle
from models.ssd_300 import SSD300
from utils.ssd_utils import *
from ssd_code.train import train

PiecewiseConstantDecay  = tf.keras.optimizers.schedules.PiecewiseConstantDecay
SGD                     = tf.keras.optimizers.SGD

# Configure object detection
train_model     = False # otherwise load model
train_oracle    = False
evaluate_oracle = False
oracle_sample   = False
visualize_model = False
calc_confidence = True

# Initialize Model
input_shape = (300, 300, 3)
coco_train, coco_validation = tfds.load("coco/2017", split=["train", "validation"], shuffle_files=True) #/mnt/storage/tfds_data
model = SSD300(num_class=80, input_shape=input_shape)
model(tf.ones([1, input_shape[0], input_shape[1], input_shape[2]], tf.int16))

#validation size = 5000
split_size = 4000
coco_val = coco_validation.take(split_size)
coco_sample = coco_validation.skip(split_size)

# model.build((None, input_shape[0], input_shape[1], input_shape[2]))
# model.summary()

# Train or load model
if train_model:
    # Training params
    lr_decay = PiecewiseConstantDecay(boundaries=[80000, 10000, 120000], values=[0.001, 0.0005, 0.0001, 0.00005])
    optimizer = SGD(learning_rate=lr_decay, momentum=0.9)

    # Output path
    now = datetime.now()
    dt_str = now.strftime("%d-%m-%Y_%H-%M-%S")
    out_path = os.path.join(os.getcwd(), 'weights', dt_str)
    os.makedirs(out_path, exist_ok=True)

    # Train
    train(model, optimizer, coco_train, out_path, n_imgs=1000, batch_size=20, n_epoch=20, inter_save=1)
else:
    #23-02-2023_16-41-15
    #28-02-2023_11-19-13
    out_path = os.path.join(os.getcwd(), "weights", "23-02-2023_16-41-15", "ssd_weights_epoch_001.h5")
    model.load_weights(out_path)
    _, default_boxes = model.get_default_boxes()
    oracle = Oracle(model)

    if train_oracle:
        
        oracle_img_n = 500
        batch_size = 20
        ds_validation = batch_dataset(coco_val.take(oracle_img_n), batch_size)
        loss = []
        imgs = []
        for batch in tqdm(ds_validation, desc="Calculating loss", position=0):
            images, scores_gt, offsets_gt, images_ref = dataset_to_ground_truth(batch, default_boxes)
            scores, offsets = model(images)
            loss.append(model.get_loss(scores, scores_gt, offsets, offsets_gt))
            imgs.append(images)

        loss = tf.concat(loss, axis=0)
        imgs = tf.concat(imgs, axis=0)
        loss = loss / 0.19358322
        # print(tf.reduce_mean(loss))

        oracle.fit(imgs, loss, epochs=10, batch_size=batch_size)
        oracle.save_weights(f"{oracle_img_n}test_oracle_weights.h5")
        # oracle.evaluate(imgs, loss)
    elif evaluate_oracle: 
        oracle.build((None, input_shape[0], input_shape[1], input_shape[2]))
        oracle.load_weights("500_oracle_weights.h5")

        # Get predictions and loss for first batch
        images, scores_gt, offsets_gt, images_ref = dataset_to_ground_truth(coco_validation.take(50), default_boxes)
        scores, offsets = model(images)

        # Evaluate first batch on untrained model
        loss = model.get_loss(scores, scores_gt, offsets, offsets_gt)
        y_gt = loss / 0.19358322
        y    = oracle(images)
        oracle.evaluate(images, y_gt)

        # Sanity check inference
        sanity_loss = tf.keras.losses.mean_absolute_error(y_gt, y)
        print(sanity_loss.numpy())

    elif oracle_sample: # Get sample from oracle for new training
        oracle.build((None, input_shape[0], input_shape[1], input_shape[2]))
        oracle.load_weights("500test_oracle_weights.h5")

        n_imgs = 50
        n_sample = 50

        dataset = coco_sample.take(n_imgs)
        data_sample = get_sample_oracle(oracle, dataset, n_sample, default_boxes)

        new_dataset = coco_train.concatenate(data_sample) # adds data_sample to end of coco_train

        # ----------------------------- NEW TRAINING -----------------------------

        # Training params
        lr_decay = PiecewiseConstantDecay(boundaries=[80000, 10000, 120000], values=[0.001, 0.0005, 0.0001, 0.00005])
        optimizer = SGD(learning_rate=lr_decay, momentum=0.9)

        # Output path
        now = datetime.now()
        dt_str = now.strftime("NEW%d-%m-%Y_%H-%M-%S")
        out_path = os.path.join(os.getcwd(), 'weights', dt_str)
        os.makedirs(out_path, exist_ok=True)

        new_model = SSD300(num_class=80, input_shape=input_shape)
        new_model(tf.ones([1, input_shape[0], input_shape[1], input_shape[2]], tf.int16))

        # Train
        new_dataset = coco_train.take(1000).concatenate(data_sample)
        train(new_model, optimizer, new_dataset, out_path, n_imgs=1000+n_sample, batch_size=10, n_epoch=20, inter_save=1)


# Validation
if visualize_model:

    label_names         = open("dataset_labels/coco_class_labels.txt").read().splitlines()
    batch_0             = batch_dataset(coco_validation, batch=5)[0]
    _, default_boxes    = model.get_default_boxes()
    visualize_ground_truth(label_names=label_names, dataset=batch_0, default_boxes=default_boxes)
    # visualize_dataset(label_names, dataset, default_boxes)
    # visualize_model_output(input_shape, model, label_names, dataset)

# Confidence
if calc_confidence:
    n_sample = 50
    batch = 5
    data_sample, img_sort_order, img_uncertainty = get_sample_confidence(model, coco_validation, n_sample, batch)

    batches = batch_dataset(coco_validation, batch=batch)
    imgs_ref = []
    for batch in tqdm(batches, desc="Extracting images", position=0):
        images, bboxes, labels             = get_coco_raw(batch)
        images, _, _, images_ref = preprocess_coco(images, bboxes, labels)
        imgs_ref.append(images_ref)

    imgs_ref = [img_ref for sublist in imgs_ref for img_ref in sublist] # flatten image list

    for idx in img_sort_order[:10]: 
        img = imgs_ref[idx]
        img = Image.fromarray(img.numpy())
        img.show()
        print(img_uncertainty[idx])
    