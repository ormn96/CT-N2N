import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

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
    [image, ] = tf.py_function(ct_intensity_to_HU, [image], [tf.int16])
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
    w = noise_gen(img, db).astype(np.int16)
    noisy_img = img + w
    return noisy_img


def augment_train(image):
    noise_db = 10.0
    noisy_1 = addNoise(image, noise_db, wgn)
    noisy_2 = addNoise(image, noise_db, wgn)
    return noisy_1, noisy_2


def augment_val(image):
    noise_db = 10.0
    clean = image
    noisy = addNoise(image, noise_db, wgn)
    return noisy, clean


def create_train_dataset(dataset_path, batch_size):
    print('Setting up training dataset source from', dataset_path)
    num_threads = tf.data.experimental.AUTOTUNE
    buf_size = 100

    list_ds = tf.data.Dataset.list_files(str(dataset_path + '/*'))

    image_ds = list_ds.map(read_image,num_parallel_calls=num_threads)

    duplicated_ds = image_ds.flat_map(dup_ds)

    augmented_ds = duplicated_ds.map(augment_train,num_parallel_calls=num_threads)

    shuffled_ds = augmented_ds.shuffle(buffer_size=buf_size)

    batched_ds = shuffled_ds.batch(batch_size)

    return batched_ds.prefetch(tf.data.experimental.AUTOTUNE)


def create_val_dataset(dataset_path, batch_size):
    print('Setting up validation dataset source from', dataset_path)
    num_threads = tf.data.experimental.AUTOTUNE
    buf_size = 100

    list_ds = tf.data.Dataset.list_files(str(dataset_path + '/*'))

    image_ds = list_ds.map(read_image,num_parallel_calls=num_threads)

    augmented_ds = image_ds.map(augment_val,num_parallel_calls=num_threads)

    shuffled_ds = augmented_ds.shuffle(buffer_size=buf_size)

    batched_ds = shuffled_ds.batch(batch_size)

    return batched_ds.prefetch(tf.data.experimental.AUTOTUNE)

