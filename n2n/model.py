from keras.models import Model
from keras.layers import Input, Add, PReLU, Conv2DTranspose, Concatenate, MaxPooling2D, UpSampling2D, Dropout
from keras.layers.convolutional import Conv2D
from tensorflow.keras.layers import BatchNormalization
from keras.callbacks import Callback
from keras import backend as K
import tensorflow as tf
import numpy as  np


def tf_log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def PSNR(y_true, y_pred):
    max_pixel = np.iinfo(np.int16).max - np.iinfo(np.int16).min
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))


# def PSNR(y_true, y_pred):
#     max_pixel = 255.0
#     y_pred = K.clip(y_pred, 0.0, 255.0)
#     return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))


# UNet: code from https://github.com/pietz/unet-keras
# names added
def get_unet_model(input_channel_num=1, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu',
         dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=False):
    def _conv_block(m, dim, acti, bn, res,name_post, do=0):
        n = Conv2D(dim, 3, activation=acti, padding='same', name=f"conv_{name_post}_1")(m)
        n = BatchNormalization()(n) if bn else n
        n = Dropout(do, name=f"dropout_{name_post}")(n) if do else n
        n = Conv2D(dim, 3, activation=acti, padding='same', name=f"conv_{name_post}_2")(n)
        n = BatchNormalization()(n) if bn else n

        return Concatenate()([m, n]) if res else n

    def _level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
        if depth > 0:
            n = _conv_block(m, dim, acti, bn, res, f"down{depth}")
            m = MaxPooling2D(name=f"maxpool_{depth}")(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
            m = _level_block(m, int(inc * dim), depth - 1, inc, acti, do, bn, mp, up, res)
            if up:
                m = UpSampling2D(name=f"upsampling_{depth}")(m)
                m = Conv2D(dim, 2, activation=acti, padding='same', name=f"upconv_{depth}")(m)
            else:
                m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
            n = Concatenate(name=f"concat_{depth}")([n, m])
            m = _conv_block(n, dim, acti, bn, res, f"up{depth}")
        else:
            m = _conv_block(m, dim, acti, bn, res, "bottom", do)

        return m

    i = Input(shape=(None, None, input_channel_num))
    o = _level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
    o = Conv2D(out_ch, 1, name=f"conv_output")(o)
    model = Model(inputs=i, outputs=o, name="encoder_decoder")

    return model


def main():
    model = get_unet_model()
    model.summary()


if __name__ == '__main__':
    main()
