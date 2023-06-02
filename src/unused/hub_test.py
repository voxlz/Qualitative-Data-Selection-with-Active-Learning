import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

from ssd_code.eval import get_predictions

# https://tfhub.dev/tensorflow/efficientdet/lite4/feature-vector/2

coco_validation: tf.data.Dataset = tfds.load("coco/2017", split="validation", shuffle_files=True) #/mnt/storage/tfds_data

a = next(iter(coco_validation))

@tf.function
def coco_preprocess(data):
    image: tf.Tensor = tf.image.resize_with_crop_or_pad(data['image'], 512, 512) # resize to 512x512
    image = image * 2 / 255 - 1 # range [-1, 1]
    return image

coco_validation = coco_validation.map(coco_preprocess).batch(32)

a = next(iter(coco_validation))
res = tf.identity(a, name="inputs")
images = res  # A batch of preprocessed images with shape [batch_size, height, width, 3].
base_model = hub.KerasLayer("https://tfhub.dev/tensorflow/efficientdet/lite4/feature-vector/1")


scores_boxes, offsets_boxes = base_model(images, training=False)
# scores  = tf.concat(scores_boxes,  axis=1)
# offsets = tf.concat(offsets_boxes, axis=1)

get_predictions(scores_boxes, offsets_boxes)

print("finished")