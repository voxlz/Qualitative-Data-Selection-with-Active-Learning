import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

from ssd_code.eval import draw_bboxes

# https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1

# Code to perform inference on a pretrained object detection model. Not really useful for our purposes, but it's a good example of how to use the model.

coco_validation: tf.data.Dataset = tfds.load("coco/2017", split="validation", shuffle_files=True) #/mnt/storage/tfds_data

a = next(iter(coco_validation))

@tf.function
def coco_preprocess(data):
    image: tf.Tensor = tf.image.resize_with_crop_or_pad(data['image'], 640, 640) # resize to 640x640
    return image

coco_validation = coco_validation.map(coco_preprocess).batch(1)

detector = hub.load("https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1")

for _ in range(5):
    image = next(iter(coco_validation))
    detector_output = detector(image)
    label_names     = open("dataset_labels/coco_class_labels - paper.txt").read().splitlines()

    boxes   = detector_output["detection_boxes"]
    classes = detector_output["detection_classes"]
    scores  = detector_output["detection_scores"]
    
    mask = scores > 0.5
    
    boxes   = tf.ragged.boolean_mask(boxes, mask, name='boolean_mask')
    classes = tf.ragged.boolean_mask(classes, mask, name='boolean_mask')
    scores  = tf.ragged.boolean_mask(scores, mask, name='boolean_mask')

    draw_bboxes(image, boxes, classes, label_names, scores, img_n = 1)


print("finished")