import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import GaussianNoise

np.set_printoptions(precision=4)


def show(image):
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.axis('off')


def ct_intensity_to_HU(image):
    return image.numpy().astype(np.float32, copy=False) - 32768


def read_image(filename):
    image = tf.io.read_file(filename, name="read_image")
    image = tf.io.decode_png(image, dtype=tf.uint16, name="decode_image")
    [image, ] = tf.py_function(ct_intensity_to_HU, [image], [tf.float32], name="convert_to_hu")
    return image


def dup_ds(image):
    return tf.data.Dataset.from_tensors(image).repeat(2)


def wgn(shape, std):
    return np.random.normal(0.0, std, shape)


def addNoise(image, std):
    n = tf.random.normal(shape=image.shape, mean=0.0, stddev=std, name="noise_gen")
    return tf.add(image, n, name="noise_add")


def create_train_dataset(dataset_path, batch_size, noise_std):
    print('Setting up training dataset source from', dataset_path)
    num_threads = tf.data.experimental.AUTOTUNE
    buf_size = 100
    noise_adder = GaussianNoise(noise_std)

    def augment_train(image):
        noisy_1 = noise_adder(image, training=True)
        noisy_2 = noise_adder(image, training=True)
        return noisy_1, noisy_2

    list_ds = tf.data.Dataset.list_files(str(dataset_path + '/*'))

    image_ds = list_ds.map(read_image, num_parallel_calls=num_threads)

    # duplicated_ds = image_ds.flat_map(dup_ds)
    # augmented_ds = duplicated_ds.map(augment_train, num_parallel_calls=num_threads)

    augmented_ds = image_ds.map(augment_train, num_parallel_calls=num_threads)

    #shuffled_ds = augmented_ds.shuffle(buffer_size=buf_size)
    #batched_ds = shuffled_ds.batch(batch_size)

    batched_ds = augmented_ds.batch(batch_size)

    return batched_ds.prefetch(tf.data.experimental.AUTOTUNE)


def create_val_dataset(dataset_path, batch_size, noise_std):
    print('Setting up validation dataset source from', dataset_path)
    num_threads = tf.data.experimental.AUTOTUNE
    buf_size = 100
    noise_adder = GaussianNoise(noise_std)

    def augment_val(image):
        clean = image
        noisy = noise_adder(image, training=True)
        return noisy, clean

    list_ds = tf.data.Dataset.list_files(str(dataset_path + '/*'))

    image_ds = list_ds.map(read_image, num_parallel_calls=num_threads)

    augmented_ds = image_ds.map(augment_val, num_parallel_calls=num_threads)

    #shuffled_ds = augmented_ds.shuffle(buffer_size=buf_size)
    #batched_ds = shuffled_ds.batch(batch_size)

    batched_ds = augmented_ds.batch(batch_size)

    return batched_ds.prefetch(tf.data.experimental.AUTOTUNE)
