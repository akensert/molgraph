import tensorflow as tf
from tensorflow import keras

from typing import Optional

from molgraph.internal import register_keras_serializable 


class MaskedLoss(keras.losses.Loss):

    'Base class for masked losses.'

    def __init__(
        self,
        reduction: str = keras.losses.Reduction.AUTO,
        name: Optional[str] = None
    ) -> None:
        super().__init__(
            reduction=keras.losses.Reduction.NONE,
            name=name,
        )
        self._reduction = reduction
        keras.losses.Reduction.validate(self._reduction)

    def __call__(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: Optional[tf.Tensor] = None
    ) -> tf.Tensor:

        losses = super().__call__(y_true, y_pred, None)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.bool)
            losses = tf.ragged.boolean_mask(losses, sample_weight)
            divisor = tf.cast(losses.row_lengths(), losses.dtype)
            losses = tf.reduce_sum(losses, axis=-1)
            losses = tf.math.divide_no_nan(losses, divisor)

        #self.reduction = self._reduction
        #self.reduction = self._get_reduction()

        if self._reduction == keras.losses.Reduction.NONE:
            return losses

        if self._reduction == keras.losses.Reduction.SUM:
            return tf.reduce_sum(losses)

        return tf.reduce_mean(losses)

    def get_config(self):
        base_config = super().get_config()
        base_config.update({'reduction': self._reduction})
        return base_config


@register_keras_serializable(package='molgraph.losses')
class MaskedBinaryCrossentropy(MaskedLoss):
    
    '''Masked binary crossentropy loss. 
    
    Useful for multi-label classification with missing labels.
    '''

    def __init__(
        self,
        gamma: float = 0.0,
        from_logits: bool = False,
        label_smoothing: float = 0.,
        reduction: str = keras.losses.Reduction.AUTO,
        name: str = 'masked_bce'
    ):
        super().__init__(
            reduction=reduction,
            name=name
        )
        self._gamma = gamma
        self._from_logits = from_logits
        self._label_smoothing = label_smoothing

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        label_smoothing = tf.convert_to_tensor(
            self._label_smoothing, dtype=y_pred.dtype)
        gamma = tf.convert_to_tensor(self._gamma, dtype=tf.float32)

        def smooth_labels():
            return y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

        y_true = tf.cond(label_smoothing != 0., smooth_labels, lambda: y_true)

        return tf.cond(
            gamma != 0.0,
            lambda: keras.backend.binary_focal_crossentropy(
                y_true, y_pred, gamma=gamma, from_logits=self._from_logits),
            lambda: keras.backend.binary_crossentropy(
                y_true, y_pred, from_logits=self._from_logits))

    def get_config(self):
        base_config = super().get_config()
        base_config.update({
            'gamma': self._gamma,
            'from_logits': self._from_logits,
            'label_smoothing': self._label_smoothing,
        })
        return base_config


@register_keras_serializable(package='molgraph.losses')
class MaskedHuber(MaskedLoss):

    '''Masked huber loss. 
    
    Useful for multi-label regression with missing labels.
    '''

    def __init__(
        self,
        delta: float = 1.0,
        reduction: str = keras.losses.Reduction.AUTO,
        name: str = 'masked_huber'
    ):
        super().__init__(
            reduction=reduction,
            name=name
        )
        self._delta = delta

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_pred = tf.cast(y_pred, dtype=keras.backend.floatx())
        y_true = tf.cast(y_true, dtype=keras.backend.floatx())
        delta = tf.cast(self._delta, dtype=keras.backend.floatx())

        error = tf.subtract(y_pred, y_true)
        abs_error = tf.abs(error)
        half = tf.convert_to_tensor(0.5, dtype=abs_error.dtype)
        return tf.where(
            abs_error <= delta,
            half * tf.square(error),
            delta * abs_error - half * tf.square(delta)
        )

    def get_config(self):
        base_config = super().get_config()
        base_config.update({'delta': self._delta})
        return base_config


@register_keras_serializable(package='molgraph.losses')
class MaskedMeanSquaredError(MaskedLoss):

    '''Masked mean squared error loss. 
    
    Useful for multi-label regression with missing labels.
    '''

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        return tf.math.squared_difference(y_pred, y_true)


@register_keras_serializable(package='molgraph.losses')
class MaskedMeanAbsoluteError(MaskedLoss):

    '''Masked mean absolute error loss. 
    
    Useful for multi-label regression with missing labels.
    '''

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        return tf.abs(y_pred - y_true)
