import tensorflow as tf
from tensorflow import keras


@keras.utils.register_keras_serializable(package='molgraph.losses')
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
        negative_score = tf.reshape(negative_score, (-1, 1))
        y_pred = tf.concat([
            positive_score, negative_score
        ], axis=0)
        y_true = tf.concat([
            tf.ones_like(positive_score), tf.zeros_like(negative_score)
        ], axis=0)
        return super().call(y_true, y_pred)


@keras.utils.register_keras_serializable(package='molgraph.losses')
class LinkContrastiveMarginLoss(keras.losses.Loss):

    def __init__(
        self,
        margin=1.,
        reduction=keras.losses.Reduction.AUTO,
        name='link_contrastive_margin_loss',
    ):
        super().__init__(reduction=reduction, name=name,)
        self.margin = margin

    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.math.maximum(0., 1. - y_true + y_pred))

    def get_config(self):
        base_config = super().get_config()
        base_config.update({'margin': self.margin})
        return base_config

@keras.utils.register_keras_serializable(package='molgraph.losses')
class LinkContrastiveBinaryCrossentropy(keras.losses.Loss):

    def __init__(
        self,
        reduction=keras.losses.Reduction.AUTO,
        name='link_contrastive_bce_loss',
    ):
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true, y_pred):
        epsilon = keras.backend.epsilon()
        y_true = tf.clip_by_value(tf.nn.sigmoid(y_true), epsilon, 1.0 - epsilon)
        y_pred = tf.clip_by_value(tf.nn.sigmoid(y_pred), epsilon, 1.0 - epsilon)
        return -tf.reduce_mean(tf.math.log(y_true) + tf.math.log(1 - y_pred))
