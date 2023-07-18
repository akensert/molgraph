import tensorflow as tf
from tensorflow import keras

from molgraph.internal import register_keras_serializable 


@register_keras_serializable(package='molgraph.losses')
class LinkBinaryCrossentropy(keras.losses.BinaryCrossentropy):

    def __init__(
        self,
        from_logits=True,
        label_smoothing=0.,
        axis=-1,
        reduction=keras.losses.Reduction.AUTO,
        name='link_bce_loss'
    ):
        super().__init__(
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            axis=axis,
            reduction=reduction,
            name=name,
        )

    def call(self, positive_score, negative_score):
        y_pred = tf.concat([
            positive_score, negative_score
        ], axis=0)
        y_true = tf.concat([
            tf.ones_like(positive_score), tf.zeros_like(negative_score)
        ], axis=0)
        return super().call(y_true, y_pred)


# TODO: Make it work for len(y_true) != len(y_pred)
@register_keras_serializable(package='molgraph.losses')
class LinkContrastiveMarginLoss(keras.losses.Loss):

    def __init__(
        self,
        margin=1.,
        reduction=keras.losses.Reduction.AUTO,
        name='link_contrastive_margin_loss',
    ):
        super().__init__(reduction=reduction, name=name,)
        self.margin = margin

    def call(self, positive_score, negative_score):
        return tf.reduce_mean(tf.math.maximum(0., 1. - positive_score + negative_score))

    def get_config(self):
        base_config = super().get_config()
        base_config.update({'margin': self.margin})
        return base_config


# TODO: Make it work for len(y_true) != len(y_pred)
@register_keras_serializable(package='molgraph.losses')
class LinkContrastiveBinaryCrossentropy(keras.losses.Loss):

    def __init__(
        self,
        reduction=keras.losses.Reduction.AUTO,
        name='link_contrastive_bce_loss',
    ):
        super().__init__(reduction=reduction, name=name)

    def call(self, positive_score, negative_score):
        epsilon = keras.backend.epsilon()
        positive_score = tf.clip_by_value(
            tf.nn.sigmoid(positive_score), epsilon, 1.0 - epsilon)
        negative_score = tf.clip_by_value(
            tf.nn.sigmoid(negative_score), epsilon, 1.0 - epsilon)
        return -tf.reduce_mean(tf.math.log(positive_score) + tf.math.log(1 - negative_score))
