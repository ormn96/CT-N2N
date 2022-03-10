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
    noisy_1.set_shape([512, 512])
    noisy_2 = addNoise(image, noise_db, wgn)
    noisy_2.set_shape([512, 512])
    return ((noisy_1, noisy_2), clean)


def create_train_dataset(dataset_path, minibatch_size):
    print('Setting up training dataset source from', dataset_path)
    num_threads = 2
    buf_size = 100

    list_ds = tf.data.Dataset.list_files(str(dataset_path + '/*'))

    image_ds = list_ds.map(read_image,num_parallel_calls=num_threads)

    duplicated_ds = image_ds.flat_map(dup_ds)

    augmented_ds = duplicated_ds.map(augment,num_parallel_calls=num_threads)

    shuffled_ds = augmented_ds.shuffle(buffer_size=buf_size)

    batched_ds = shuffled_ds.batch(minibatch_size)

    return batched_ds


def create_val_dataset(dataset_path, minibatch_size):
    print('Setting up validation dataset source from', dataset_path)
    num_threads = 2
    buf_size = 100

    list_ds = tf.data.Dataset.list_files(str(dataset_path + '/*'))

    image_ds = list_ds.map(read_image,num_parallel_calls=num_threads)

    augmented_ds = image_ds.map(augment,num_parallel_calls=num_threads)

    shuffled_ds = augmented_ds.shuffle(buffer_size=buf_size)

    batched_ds = shuffled_ds.batch(minibatch_size)

    return batched_ds