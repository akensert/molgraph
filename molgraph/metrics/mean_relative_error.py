import tensorflow as tf
from tensorflow import keras

from molgraph.internal import register_keras_serializable 


@register_keras_serializable(package='molgraph.metrics')
class MeanRelativeError(keras.metrics.Mean):

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)
        abs_error = tf.abs(y_true - y_pred)
        mr_error = tf.math.divide_no_nan(abs_error, tf.math.abs(y_true))
        return super().update_state(mr_error, sample_weight=sample_weight)
