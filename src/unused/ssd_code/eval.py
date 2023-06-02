from PIL import Image, ImageDraw
import random
import tensorflow as tf

from utils.ssd_utils import bboxes_center_to_corner, get_coco_raw, dataset_to_ground_truth, preprocess_coco

# TODO: Class specific colors?
colors = [
    (random.randint(50, 200), random.randint(50, 200), random.randint(100, 200))
    for _ in range(50)
]

def get_predictions(scores, offsets, default_boxes, score_threshold = 0.2, max_output = 200, iou_threshold = 0.45):
    ''' Filters model output and processes it to be visualize.
    
    Returns:
    - labels [>200,]
    - bboxes [>200, 4]
    - scores [>200,]
    '''
    assert scores.dtype == tf.float32
    
    p_labels = []
    p_boxes = []
    p_scores = []
    # Filter scores and offsets based on score_threshold
    for img_scores, img_offsets in zip(scores, offsets):
        
        # Reduce dimension and find mask
        max_score       = tf.reduce_max(img_scores, axis=1)
        labels          = tf.argmax(img_scores, axis=1)
        score_threshold = tf.convert_to_tensor(score_threshold, tf.float32)
        mask            = max_score >= score_threshold
        
        # Calculate bounding boxes and filter 
        img_boxes  = default_boxes[mask] + img_offsets[mask]
        img_boxes  = bboxes_center_to_corner(img_boxes)
        img_scores = max_score[mask]
        img_labels = labels[mask]
        
        # NMS
        selected   = tf.image.non_max_suppression(img_boxes, img_scores, max_output_size=max_output, iou_threshold=iou_threshold, score_threshold=score_threshold)
        img_boxes  = tf.gather(img_boxes, selected)
        img_scores = tf.gather(img_scores, selected)
        img_labels = tf.gather(img_labels, selected)
        
        p_boxes.append(img_boxes)
        p_scores.append(img_scores)
        p_labels.append(img_labels)
        
    # Return labels, boxes and scores?
    return p_labels, p_boxes, p_scores

def draw_bboxes(imgs, bboxes, labels, label_names, scores, img_n = 5):
    ''' 
    Visualizes bounding boxes, labels and scores on top the image. Images may be of different sizes. 
    
    Input: 
        - imgs: [N, Tensor(w, h, c)] unnormalized
        - bboxes: [N, Tensor(objs, 4)] NORMALIZED (ymin, xmin, ymax, xmax) 
        - labels: [N, Tensor(objs)]
        - label_names: [C]
        - scores: [N, Tensor(objs)]
        - img_n: Number of images to print
    '''
    for img_idx, image in enumerate(imgs):
        if img_idx >= img_n: break
        img = Image.fromarray(image.numpy())
        draw = ImageDraw.Draw(img)
        for box_idx, bbox in enumerate(bboxes[img_idx]):
            img_w      = image.shape[1]
            img_h      = image.shape[0]
            color      = random.choice(colors)
            top_left   = (bbox[1]*img_w, bbox[0]*img_h)
            btm_right  = (bbox[3]*img_w, bbox[2]*img_h)
            box_label  = labels[img_idx][box_idx]
            box_label  = int(box_label.numpy())
            class_name = label_names[box_label]
            confidence = scores[img_idx][box_idx]
            draw.rectangle([top_left, btm_right], outline=color)
            draw.text(top_left, "%s (%.2f)" % (class_name, confidence), fill=color)
        img.show()
        
# Visualize dataset
def visualize_dataset(label_names, dataset, default_boxes):
    imgs, bboxes, labels, _ = get_coco_raw(dataset)
    scores                  = tf.ones([len(imgs), len(default_boxes)], tf.uint8)
    draw_bboxes(imgs, bboxes, labels, label_names, scores)

# Extract ground truth
def visualize_ground_truth(label_names, dataset, default_boxes):
    _, scores, offsets, imgs_cropped = dataset_to_ground_truth(dataset, default_boxes)
    labels, boxes, scores            = get_predictions(scores, offsets, default_boxes)
    draw_bboxes(imgs_cropped, boxes, labels, label_names, scores)
    
# Visualize model output
def visualize_model_output(input_shape, model, label_names, dataset):
    imgs, bboxes, labels, _             = get_coco_raw(dataset)
    imgs, bboxes, labels, imgs_cropped  = preprocess_coco(imgs, bboxes, labels, input_shape)
    _, default_boxes                    = model.get_default_boxes()
    scores, offsets                     = model(tf.convert_to_tensor(imgs))
    scores                              = tf.math.softmax(scores, axis=2)
    labels, boxes, scores               = get_predictions(scores, offsets, default_boxes)
    draw_bboxes(imgs_cropped, boxes, labels, label_names, scores)