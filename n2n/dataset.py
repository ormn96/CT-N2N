import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=4)


def show(image):
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.axis('off')


# CT need to be transformed from uint16 to int16
# HU values are from [-1204 : 3071]
def ct_intensity_to_HU(image):
    im = image.numpy().astype(np.float32, copy=False) - 32768
    return (im - (-1024)).astype(np.int16, copy=False)


def read_image(filename, image_size):
    image = tf.io.read_file(filename, name="read_image")
    image = tf.io.decode_png(image, dtype=tf.uint16, name="decode_image")
    image = tf.image.resize_with_crop_or_pad(image, image_size, image_size)
    [image, ] = tf.py_function(ct_intensity_to_HU, [image], [tf.int16], name="convert_to_hu")
    return image


def dup_ds(image):
    return tf.data.Dataset.from_tensors(image).repeat(2)


# std in HU values is treated as float and then transformed to int6 in the hu range(4096)
def addNoise(image, std):
    n = tf.random.normal(shape=image.shape, mean=0.0, stddev=std, name="noise_gen") * 4096
    return tf.add(image, tf.cast(n, tf.int16), name="noise_add")


def create_train_dataset(dataset_path, batch_size, noise_std, image_size):
    print('Setting up training dataset source from', dataset_path)
    num_threads = tf.data.experimental.AUTOTUNE

    def augment_train(image):
        [noisy_1, ] = tf.py_function(addNoise, [image, noise_std], [tf.int16], name="noise_add_train_1")
        [noisy_2, ] = tf.py_function(addNoise, [image, noise_std], [tf.int16], name="noise_add_train_2")
        return noisy_1, noisy_2

    def _read_image(filename):
        return read_image(filename, image_size)

    list_ds = tf.data.Dataset.list_files(str(dataset_path + '/*'))

    image_ds = list_ds.map(_read_image, num_parallel_calls=num_threads)

    # duplicated_ds = image_ds.flat_map(dup_ds)
    # augmented_ds = duplicated_ds.map(augment_train, num_parallel_calls=num_threads)

    augmented_ds = image_ds.map(augment_train, num_parallel_calls=num_threads)

    batched_ds = augmented_ds.batch(batch_size)

    return batched_ds.prefetch(tf.data.experimental.AUTOTUNE)


def create_val_dataset(dataset_path, batch_size, noise_std, image_size):
    print('Setting up validation dataset source from', dataset_path)
    num_threads = tf.data.experimental.AUTOTUNE

    def augment_val(image):
        clean = image
        [noisy, ] = tf.py_function(addNoise, [image, noise_std], [tf.int16], name="noise_add_val")
        return noisy, clean

    def _read_image(filename):
        return read_image(filename, image_size)

    list_ds = tf.data.Dataset.list_files(str(dataset_path + '/*'))

    image_ds = list_ds.map(_read_image, num_parallel_calls=num_threads)

    augmented_ds = image_ds.map(augment_val, num_parallel_calls=num_threads)

    batched_ds = augmented_ds.batch(batch_size)

    return batched_ds.prefetch(tf.data.experimental.AUTOTUNE)
