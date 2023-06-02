import json
import tensorflow as tf


def tensor_equal(a: tf.Tensor, b: tf.Tensor):
    ''' Check if all elements in two tensors are equal '''
    return tf.math.reduce_all(tf.math.equal(a, b))

def tensor_set_equal(a, b):
    ''' Check that two lists of tensors have the same content, not necessarily in the same order '''
    xs = [[tensor_equal(a_i, b_i) for a_i in a] for b_i in b] # Check all tensors against each other
    xs = [any(x) for x in xs] # Check if the "a" tensors match any of the "b" tensors 
    return all(xs) # make sure all match

@tf.function
def normalize_oracle_batch_input(img, loss):
    ''' Calculate the mean and variance of the batched loss tensor, then normalize the loss tensor'''
    mean, variance = tf.nn.moments(loss, 0)
    batch = tf.nn.batch_normalization(loss, mean, variance, variance_epsilon=1e-3, offset=None, scale=None)
    return img, batch

def imagenet_decode_predictions(y_pred, top=5):
    """Modified decoder from keras.applications.imagenet_utils
    
    Decodes the prediction of an ImageNet model.

    Args:
      y_pred: Numpy array encoding a batch of predictions.
      top: Integer, how many top-guesses to return. Defaults to 5.

    Returns:
      A list of lists of top class prediction tuples
      `(class_name, class_description, score, class_id)`.
      One list of tuples per sample in batch input.

    Raises:
      ValueError: In case of invalid shape of the `pred` array
        (must be 2D).
    """
    if len(y_pred.shape) != 2 or y_pred.shape[1] != 1000:
        raise ValueError(
            f"`decode_predictions` expects a batch of predictions (i.e. a 2D array of shape (samples, 1000)). Found array with shape: {str(y_pred.shape)}"
        )
    CLASS_INDEX = None
    if CLASS_INDEX is None:
        with open("dataset_labels/imagenet_class_index.json") as f:
            CLASS_INDEX = json.load(f)
    results = []
    for pred in y_pred:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i], i) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results