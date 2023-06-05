from collections import Counter
from itertools import accumulate
from datasets import load_dataset
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

from vgg16.vgg16_utils import preprocess_CORRUPT_imagenet, preprocess_imagenet, randomly_augment, tupleize
from vgg16.vgg16_datamix import Datamix
layers  = tf.keras.layers
ClassLabel = tfds.features.ClassLabel
Dataset = tf.data.Dataset

def get_dataset(conf):
    ''' Get dataset from tfds or hugging face.'''
    
    # Calculate splits
    # Test and val first, to ensure changing size of base or unseen does not affect them.
    split_nbs = [conf['n_test'], conf['n_vali'], conf['n_unseen'], conf['n_base']] 

    # Configure dataset
    s = list(accumulate(split_nbs))
    d = conf['datamix']
    if d == Datamix.imageNet:
        dataset = get_imagenet_dataset(s, conf)
    elif (d == Datamix.v2):

        split = [f"test[:{s[0]}]", f"test[{s[0]}:{s[1]}]", f"test[{s[1]}:{s[2]}]"] + ([f"test[{s[2]}:{s[3]}]"] if conf['n_base'] != 0 else [])

        # ! SHUFFLE needs to be FALSE FOR REPRODUCIBILITY --- 
        dataset_v2 = tfds.load('imagenet_v2', split=split, shuffle_files=False) 

        dataset = [ds.map(
            preprocess_imagenet,                 
            num_parallel_calls=8 # Autotune seems to crash ram... Keep it fixed for now
        ) for ds in dataset_v2]

    elif (d == Datamix.hard):
        dataset = get_hard_dataset(split_nbs, conf)
    else:
        dataset = get_other_dataset(split_nbs, conf, d, s)

    dataset = {
        'test':     dataset[0],
        'vali':     dataset[1],
        'unseen':   dataset[2],
        'base':     dataset[3] if conf['n_base'] > 0 else None
    }

    # Sanity check
    sanity_check_dataset(dataset)

    # Count elements in tf.data.Dataset
    if debug := True:
        debug_dataset(dataset, conf)

    return dataset, sum(split_nbs)


# TODO Rename this here and in `get_dataset`
def get_imagenet_dataset(s, conf):
    split = [f"train[:{s[0]}]", f"train[{s[0]}:{s[1]}]", f"train[{s[1]}:{s[2]}]"] + ([f"train[{s[2]}:{s[3]}]"] if conf['n_base'] != 0 else [])
    print("Loading dataset...")
    dataset = load_dataset("imagenet-1k", split=split)

    print("Converting to TF Dataset...")
    tf_ds = [
        ds.with_format("tf")
        .to_tf_dataset(batch_size=conf['n_batch'])
        .unbatch()
        for ds in dataset
    ]

    print("PreProcessing Dataset... Might take a while...")
    dataset = [
        ds.map(
            tupleize,
            num_parallel_calls=8,  # Autotune seems to crash ram... Keep it fixed for now
        )
        for ds in tf_ds
    ]

    print("Done!")

    return dataset

def get_other_dataset(split_nbs, conf, d, s):
    dataset = get_mixed_data(split_nbs)


    if (d == Datamix.mixed):
        dataset = dataset.prefetch(16).map(preprocess_imagenet, num_parallel_calls=8)
    elif (d == Datamix.duplicate):
        dataset = dataset.prefetch(16).map(preprocess_imagenet, num_parallel_calls=8)

        assert conf['n_unseen'] % 4 == 0, "n_unseen must be divisible by 4, since we use 4 duplicates of each image."
        
        # Take 1/4 of the dataset and repeat it 4 times, then add the rest of the images.
        n_unique        = conf['n_unseen']# train count
        aug_count       = int(n_unique / 4)
        training = dataset.take(aug_count).repeat(4)
        val_test_base = dataset.skip(aug_count).take(sum(split_nbs) - conf['n_unseen'])
        dataset = training.concatenate(val_test_base)
    elif (d == Datamix.noisy):
        dataset = dataset.prefetch(16).map(
            preprocess_CORRUPT_imagenet, num_parallel_calls=8
        )

    test_data = dataset.take(split_nbs[0])
    vali_data = dataset.skip(s[0]).take(split_nbs[1])
    unseen_data = dataset.skip(s[1]).take(split_nbs[2])
    base_data = dataset.skip(s[2]).take(split_nbs[3])
    dataset = [test_data, vali_data, unseen_data, base_data]

    return dataset

def get_hard_dataset(split_nbs, conf):
    result = get_mixed_data(split_nbs)

    result = result.map(preprocess_CORRUPT_imagenet, num_parallel_calls=8)

    n_unique        = conf['n_unseen'] # train count
    aug_count       = int(n_unique / 4)
    traning = (
        result.take(aug_count)
        .map(randomly_augment, num_parallel_calls=8)
        .repeat(4)
    )
    val_test_base = result.skip(aug_count)
    test_data       = val_test_base.take(conf['n_test'])
    vali_data       = val_test_base.skip(conf['n_test']).take(conf['n_vali'])
    result = [test_data, vali_data, traning]
    return result

def get_mixed_data(split_nbs):
    dataset_v2 = tfds.load('imagenet_v2', split='test[:4000]', shuffle_files=False)
    dataset_a  = tfds.load('imagenet_a' , split='test[:4000]', shuffle_files=False)
    dataset_r  = tfds.load('imagenet_r' , split='test[:4000]', shuffle_files=False)
    n_images   = sum(split_nbs)
    return Dataset.choose_from_datasets(
        [dataset_v2, dataset_a, dataset_r], Dataset.range(3).repeat()
    ).take(n_images)

def sanity_check_dataset(dataset):
    a = next(iter(dataset['unseen']))[1].numpy()
    a2 = next(iter(dataset['unseen']))[1].numpy()
    b = next(iter(dataset['test']))[1].numpy()
    c = next(iter(dataset['vali']))[1].numpy()
    assert a != b != c, "Might be from same dataset, might also just be a coincidence."
    assert a == a2, "Should always return the same value for reproduceability"

def debug_dataset(dataset, conf):
    assert dataset["unseen"].reduce(0, lambda x, _: x + 1).numpy() == conf['n_unseen'], "n_unseen is not correct"
    assert dataset["test"].reduce(0,   lambda x, _: x + 1).numpy() == conf['n_test'],   "n_test is not correct"
    assert dataset["vali"].reduce(0,   lambda x, _: x + 1).numpy() == conf['n_vali'],   "n_vali is not correct"
    assert (
        dataset["base"].reduce(0, lambda x, _: x + 1).numpy()
        if dataset["base"] is not None
        else conf['n_base'] == 0
    ), "n_base is not correct"
    
    # res = tfds.benchmark(dataset["unseen"], batch_size=1)
    # print(res)
    
    # deprocess_first_few_images(unseen_data, n_images=5)
    # deprocess_first_few_images(vali_data, n_images=5)
    
    plot_occurrences(
        "train_occurrences",
        dataset,
        'unseen',
        'Class distribution of train dataset',
        conf
    )
    plot_occurrences(
        "eval_occurrences",
        dataset,
        'test',
        'Class distribution of eval dataset',
        conf
    )
    plot_occurrences(
        "vali_occurrences",
        dataset,
        'vali',
        'Class distribution of eval dataset',
        conf
    )

def plot_occurrences(fig_name, dataset, dataset_name, title, conf):
    
    # Count occurrences of each class
    plt.figure(fig_name)
    result = Counter(list(map(lambda x: x[1], dataset[dataset_name].as_numpy_iterator())))
    plt.bar(result.keys(), result.values())
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Occurrences')
    plt.ylim(0, 100)
    plt.suptitle(f"{conf['datamix']} dataset", weight='bold')
    plt.show(block=False)

    return result