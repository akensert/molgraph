import tensorflow as tf
from tensorflow import keras

from molgraph.internal import register_keras_serializable 

from molgraph.metrics.mean_relative_error import MeanRelativeError


@register_keras_serializable(package='molgraph.metrics')
class MaskedMeanSquaredError(keras.metrics.MeanSquaredError):

    def update_state(self, y_true, y_pred, sample_weight=None):
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.bool)
            y_true = tf.ragged.boolean_mask(y_true, sample_weight)
            y_pred = tf.ragged.boolean_mask(y_pred, sample_weight)
        return super().update_state(y_true, y_pred, None)


@register_keras_serializable(package='molgraph.metrics')
class MaskedMeanAbsoluteError(keras.metrics.MeanAbsoluteError):

    def update_state(self, y_true, y_pred, sample_weight=None):
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.bool)
            y_true = tf.ragged.boolean_mask(y_true, sample_weight)
            y_pred = tf.ragged.boolean_mask(y_pred, sample_weight)
        return super().update_state(y_true, y_pred, None)


@register_keras_serializable(package='molgraph.metrics')
class MaskedMeanRelativeError(MeanRelativeError):

    def update_state(self, y_true, y_pred, sample_weight=None):
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.bool)
            y_true = tf.ragged.boolean_mask(y_true, sample_weight)
            y_pred = tf.ragged.boolean_mask(y_pred, sample_weight)
        return super().update_state(y_true, y_pred, None)


@register_keras_serializable(package='molgraph.metrics')
class MaskedRootMeanSquaredError(keras.metrics.RootMeanSquaredError):

    def update_state(self, y_true, y_pred, sample_weight=None):
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.bool)
            y_true = tf.ragged.boolean_mask(y_true, sample_weight)
            y_pred = tf.ragged.boolean_mask(y_pred, sample_weight)
        return super().update_state(y_true, y_pred, None)
