import numpy as np
import tensorflow as tf
from  PIL import Image

layers  = tf.keras.layers
Dataset = tf.data.Dataset

@tf.function
def preprocess_imagenet(data):
    img = data['image']
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.vgg16.preprocess_input(img)
    return (img, data['label'])

@tf.function
def tupleize(data):
    return (data['image'], data['label'])

def preprocess_imagenet_hugging_face(data):
    img = np.array(data['image'])

    # # Deal with the occasional grayscale image
    if len(img.shape) == 2: 
        img = np.expand_dims(img, -1)
        img = np.concatenate((img,img,img), axis=2)
        
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.vgg16.preprocess_input(img)
    img = img.numpy()
    return {'image': img, 'label': data['label']}
  
@tf.function
def preprocess_CORRUPT_imagenet(data):
    img = data['image']
    img = tf.keras.applications.vgg16.preprocess_input(img)
    img = tf.image.resize(img, (224, 224))
    return (img, data['label'])

augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
])

@tf.function
def randomly_augment(x, y):
    return (augment(x, training=True), y)
  
@tf.function
def deprocess_imagenet(img, label):
    img = tf.numpy_function(deprocess_np_img, [img], tf.uint8)
    return (img, label)

def deprocess_np_img(img):
    '''reverse preprocessing of imagenet images'''
    img = img + [103.939, 116.779, 123.68] # Undo preprocessing
    return img[:, :, ::-1].astype('uint8') # BGR -> RGB

def deprocess_first_few_images(dataset, n_images):
    '''deprocess first few images in dataset'''
    for img, label in dataset.take(n_images):
        img = deprocess_np_img(img.numpy())
        Image.fromarray(img).show()