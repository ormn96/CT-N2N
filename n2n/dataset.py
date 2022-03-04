import tensorflow as tf

import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.image as mpimg

np.set_printoptions(precision=4)


def show(image):
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.axis('off')


def ct_intensity_to_HU(image):
    return (image.numpy().astype(np.float32, copy=False) - 32768).astype(np.int16, copy=False)


def read_image(filename):
    image = tf.io.read_file(filename)
    image = tf.io.decode_png(image)
    image = tf.image.convert_image_dtype(image, tf.uint16)
    image = tf.image.resize(image, [512, 512])
    image = tf.squeeze(image)
    [image, ] = tf.py_function(ct_intensity_to_HU, [image], [tf.int16])
    image.set_shape([512, 512])
    return image


def dup_ds(image):
    return tf.data.Dataset.from_tensors(image).repeat(2)


def wgn(x, snr):
    x = tf.cast(x, tf.float32)
    snr = 10 ** (snr / 10.0)
    xpower = np.mean(x ** 2)
    npower = xpower / snr
    return np.random.randn(*x.shape.as_list()) * np.sqrt(npower)


def addNoise(img, db, noise_gen):
    w = tf.py_function(noise_gen, [img, db], [tf.float32])
    noisy_img = tf.add(img, w)
    return tf.squeeze(noisy_img)


def augment(image):
    noise_db = 10.0
    clean = image
    noisy_1 = addNoise(image, noise_db, wgn)
    noisy_2 = addNoise(image, noise_db, wgn)
    return ((noisy_1, noisy_2), clean)


def get_dataset_partitions_tf(ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True,
                              shuffle_size=1000):
    assert (train_split + test_split + val_split) == 1

    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=12)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds




def create_dataset(dataset_path, minibatch_size, add_noise):
    print('Setting up dataset source from', dataset_path)
    num_threads = 2
    buf_size = 1000

    list_ds = tf.data.Dataset.list_files(str(dataset_path + '/*'))

    image_ds = list_ds.map(read_image,num_parallel_calls=num_threads)

    duplicated_ds = image_ds.flat_map(dup_ds)

    augmented_ds = duplicated_ds.map(augment,num_parallel_calls=num_threads)

    # to delete
    augmented_ds = augmented_ds.repeat()

    shuffled_ds = augmented_ds.shuffle(buffer_size=buf_size)

    batched_ds = shuffled_ds.batch(minibatch_size)

    return batched_ds

