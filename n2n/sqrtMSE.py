from tensorflow.keras.losses import MeanSquaredError
import tensorflow as tf


class SqrtMSE(MeanSquaredError):

    def call(self, y_true, y_pred):
        return tf.math.sqrt(super(SqrtMSE, self).call(y_true, y_pred))
