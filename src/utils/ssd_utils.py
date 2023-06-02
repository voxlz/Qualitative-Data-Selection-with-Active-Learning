# Code taken from https://github.com/Apiquet/Tracking_SSD_ReID/blob/main/data_management/VOC2012ManagerObjDetection.py

import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
bw = tf.bitwise    

def bbox_corner_to_center(bbox):
    ''' Convert bbox cords from [ymin, xmin, ymax, xmax] to [cx, cy, w, h] '''
    ymin, xmin, ymax, xmax = bbox
    w  = xmax - xmin
    h  = ymax - ymin
    cx = xmin + w/2
    cy = ymin + h/2
    return tf.convert_to_tensor([cx, cy, w, h])

def bboxes_center_to_corner(boxes):
    ''' Convert bbox cords from [cx, cy, w, h] to [ymin, xmin, ymax, xmax] '''

    boxes = tf.concat([[boxes[:, 1]], [boxes[:, 0]], [boxes[:, 3]], [boxes[:, 2]]], axis=0)
    boxes = tf.transpose(boxes)
    return tf.concat([boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2], axis=-1)
    
def preprocess_coco(imgs, labels, bboxes, input_shape: tuple = (300, 300, 3)):
    '''
        Input:
            - imgs
            - bboxes (ymin, xmin, ymax, xmax)
            - labels
            - input_shape (width, hight, channels)
        Returns:
            - (List<Tensors>) images [n_img, 300, 300, 3]
            - (List<Tensors>) boxes (cx, cy, w, h)
            - (List<Tensors>) labels
            - ref_imgs
    '''
    pre_imgs = []
    pre_bboxes = []
    pre_labels = []
    ref_imgs = []

    for image, img_bboxes, img_labels in zip(imgs, bboxes, labels): 
        
        # Pad image with 300 to the right and bottom, and crop out top right corner
        img = np.pad(image.numpy(), [(0, 300), (0, 300), (0, 0)], mode='constant', constant_values=0)
        img = tf.convert_to_tensor(img) # adds zeros to dim (start, end)
        img = tf.image.crop_to_bounding_box(img, 0, 0, input_shape[0], input_shape[1])
        ref_imgs.append(img)

        # Normalize image + preprocess
        img = tf.keras.applications.vgg16.preprocess_input(img) # RGB -> BGR
        img = img / 255
        pre_imgs.append(img)

        # Revert normalize bounding boxes
        img_bboxes = img_bboxes.numpy()
        img_bboxes[:,[0,2]] *= image.shape[0]
        img_bboxes[:,[1,3]] *= image.shape[1]

        # Ensure boxes are inside the cropped image
        img_bboxes[:,[0,2]] = tf.clip_by_value(img_bboxes[:,[0,2]], 0, input_shape[0]-1)
        img_bboxes[:,[1,3]] = tf.clip_by_value(img_bboxes[:,[1,3]], 0, input_shape[1]-1)

        # Remove 0 width/hight bboxes (happens if outside of cropped image)
        if len(img_bboxes) > 0:
            is_squashed = lambda bbox: bbox[0] != bbox[2] and bbox[1] != bbox[3]
            filter_arr = np.array(list(map(is_squashed, img_bboxes)))
            img_bboxes = img_bboxes[filter_arr]
            img_labels = img_labels[filter_arr]

        # Re-normalize bounding boxes
        img_bboxes[:,[0,2]] /= input_shape[0]
        img_bboxes[:,[1,3]] /= input_shape[1]
        img_bboxes = list(map(bbox_corner_to_center, img_bboxes))

        pre_bboxes.append(tf.cast(img_bboxes, tf.float32))
        pre_labels.append(tf.cast(img_labels, tf.uint8))

    pre_imgs = tf.convert_to_tensor(pre_imgs)

    return pre_imgs, pre_labels, pre_bboxes, ref_imgs

def get_coco_raw(dataset):
    '''Extract unprocessed image and label data for later use'''

    # TODO: Filter out images with crowd labels

    # list of dicts -> dict of lists
    if (len(dataset) > 0):
        dataset = pd.DataFrame(list(dataset)).to_dict(orient="list")
        dataset['objects'] = pd.DataFrame(list(dataset['objects'])).to_dict(orient="list")

    return dataset['image'], dataset['objects']['label'], dataset['objects']['bbox']

def get_bboxes_with_overlap(gt_box: tf.Tensor, default_boxes: tf.Tensor, iou_threshold: float):
    '''
    (Speed up) Method to get the boolean tensor where iou is superior to
    the specified threshold between the gt box and the default one
    D: number of default boxes
    
    Args:
        - (tf.Tensor) box with 4 parameters: cx, cy, w, h [4]
        - (tf.Tensor) box with 4 parameters: cx, cy, w, h [D, 4]
        - (float) iou threshold to use
        
    Return:
        - (tf.Tensor) 0 if iou > threshold, 1 otherwise [D]
    '''
    # convert to xmin, ymin, xmax, ymax
    default_boxes = tf.concat(
        [
            default_boxes[:, :2] - default_boxes[:, 2:] / 2,
            default_boxes[:, :2] + default_boxes[:, 2:] / 2,
        ],
        axis=-1,
    )
    gt_box = tf.concat([gt_box[:2] - gt_box[2:] / 2, gt_box[:2] + gt_box[2:] / 2], axis=-1)
    gt_box = tf.expand_dims(gt_box, 0)
    gt_box = tf.repeat(gt_box, repeats=[default_boxes.shape[0]], axis=0)

    # compute intersection
    inter_xy_min = tf.math.maximum(default_boxes[:, :2], gt_box[:, :2])
    inter_xy_max = tf.math.minimum(default_boxes[:, 2:], gt_box[:, 2:])
    inter_width_height = tf.clip_by_value(inter_xy_max - inter_xy_min, 0.0, 300.0)
    inter_area = inter_width_height[:, 0] * inter_width_height[:, 1]

    # compute area of the boxes
    gt_box_width_height = tf.clip_by_value(gt_box[:, 2:] - gt_box[:, :2], 0.0, 300.0)
    gt_box_width_height_area = gt_box_width_height[:, 0] * gt_box_width_height[:, 1]

    default_boxes_width_height = tf.clip_by_value(
        default_boxes[:, 2:] - default_boxes[:, :2], 0.0, 300.0
    )
    default_boxes_width_height_area = (
        default_boxes_width_height[:, 0] * default_boxes_width_height[:, 1]
    )

    # compute iou
    iou = inter_area / (gt_box_width_height_area + default_boxes_width_height_area - inter_area)
    return tf.dtypes.cast(iou >= iou_threshold, tf.uint8)

def getLocOffsets(gt_box: tf.Tensor, iou_bin: tf.Tensor, default_boxes: tf.Tensor, floatType = tf.float32):
    """
    (Speed up) Method to get the offset from default boxes to box_gt on cx, cy, w, h
    where iou_idx is 1
    D: number of default boxes
    Args:
        - (tf.Tensor) box with 4 parameters: cx, cy, w, h [4]
        - (tf.Tensor) 1 if iou > threshold, 0 otherwise [D]
        - (tf.Tensor) default boxes with 4 parameters: cx, cy, w, h [D, 4]
    Return:
        - (tf.Tensor) offsets if iou_bin == 1, otherwise 0 [D, 4]
    """
    gt_box = tf.expand_dims(gt_box, 0)
    gt_box = tf.repeat(gt_box, repeats=[default_boxes.shape[0]], axis=0)
    offsets = gt_box - default_boxes

    iou_bin = tf.expand_dims(iou_bin, 1)
    iou_bin = tf.repeat(iou_bin, repeats=[4], axis=1)
    offsets = offsets * tf.dtypes.cast(iou_bin, floatType)
    return offsets

def get_coco_gt(imgs, labels, bboxes, default_boxes, n_labels=80):
    """
    Method to turn label data (boxes and labels) into ground truth data (scores and offsets)
       
    Return:
        - (tf.Tensor) images:               [Batch, 300, 300, 3]
        - (tf.Tensor) scores ground truth:  [Batch, Df_bx, Classes] 
        - (tf.Tensor) offsets ground truth: [Batch, Df_bx, 4]
    """
    gt_scores = []
    gt_offsets = []
    
    # Iterate over images
    for img_idx, img_bboxes in tqdm(enumerate(bboxes), desc="Processing ground truth data...", position=1, leave=False):
        
        # Create empty tensors of the right shape
        n_df_bxs = default_boxes.shape[0]
        img_scores = tf.zeros([n_df_bxs, n_labels], tf.float32)
        img_offsets = tf.zeros([n_df_bxs, 4], tf.float32)
        occupied_mask = tf.zeros([n_df_bxs], tf.uint8)
        
        # Iterate over boxes in image
        for box_idx, obj_bbox in enumerate(img_bboxes):
            
            # Create binary mask of overlap
            boxes_mask    = get_bboxes_with_overlap(obj_bbox, default_boxes, 0.5) # uint8
            boxes_mask    = bw.bitwise_and(boxes_mask, bw.invert(occupied_mask), "Remove occupied boxes")
            occupied_mask = bw.bitwise_or(boxes_mask, occupied_mask, "Update occupied")
            
            # Given the was a match (boxes_mask), calc diff between obj_bbox (true box) with default_boxes (quantized box)
            img_offsets = img_offsets + getLocOffsets(obj_bbox, boxes_mask, default_boxes)
            
            # Save which boxes are being used by assigning them the class label index
            one_hot    = tf.one_hot(labels[img_idx][box_idx], n_labels) if labels[img_idx].shape[0] > 0 else tf.zeros([n_labels], tf.float32)
            one_hot    = tf.expand_dims(one_hot, 0) # Needed for tf.repeat
            boxes_mask = tf.expand_dims(boxes_mask, 0) # Needed for 2d multiplication
            boxes_mask = tf.cast(boxes_mask, tf.float32)
            img_scores = img_scores + tf.transpose(boxes_mask) * one_hot

        gt_scores.append(img_scores)
        gt_offsets.append(img_offsets)

    return (
        tf.convert_to_tensor(imgs),
        tf.convert_to_tensor(gt_scores, dtype=tf.float32),
        tf.convert_to_tensor(gt_offsets, dtype=tf.float32),
    )

def dataset_to_ground_truth(dataset: tf.data.Dataset, default_boxes: list, input_shape = (300,300,3)):
    ''' Convenience function, dataset to ground truth data! '''
    imgs, labels, bboxes           = get_coco_raw(dataset)
    imgs, labels, bboxes, imgs_ref = preprocess_coco(imgs, labels, bboxes, input_shape)
    imgs, scores, offsets          = get_coco_gt(imgs, labels, bboxes, default_boxes)
    return imgs, scores, offsets, imgs_ref


def batch_dataset(dataset, batch=5):
    ''' Takes a dataset and batches through an outer dimension, resulting in [batch, data]'''
    dataset = list(dataset)
    dataset = np.asarray(dataset)
    dataset = np.reshape(dataset, (int(len(dataset) / batch), batch))
    return dataset

def get_subset_dataset(dataset, indices):
    ''' Select n_sample data points from dataset using the given indices.
            
    Returns:
    - data_samples: tf.data.Dataset with selected data points'''
    @tf.function
    def filter_fun(idx):
        compare = tf.equal(idx, indices)
        return tf.math.reduce_any(compare, -1)
        
    enum_ds = dataset.enumerate()
    data_sample = enum_ds.filter(lambda idx, _: filter_fun(idx))
    final_data_sample = data_sample.map(lambda _, value: value) # to get rid of indices again
    return final_data_sample

def get_subsets_dataset(dataset: tf.data.Dataset, indices):
    ''' Select n_sample data points from dataset using the given indices.
            
    Returns:
    - data_samples: tf.data.Dataset with selected data points
    - data_rest: tf.data.Dataset with remaining data points'''
    
    @tf.function
    def filter_fun(idx):
        compare = tf.equal(idx, indices)
        return tf.math.reduce_any(compare, -1)
    
    data_sample = dataset.enumerate().filter(lambda idx, _: filter_fun(idx))
    data_rest = dataset.enumerate().filter(lambda idx, _: not filter_fun(idx))    
    # count = data_sample.reduce(0, lambda x, _: x + 1).numpy()
    # assert count == len(indices), "Not all indices were found in the dataset! Found only %s of %s." % (count, len(indices))

    final_data_sample = data_sample.map(lambda _, value: value) # revert enumeration
    final_data_rest = data_rest.map(lambda _, value: value) # revert enumeration
    return final_data_sample, final_data_rest