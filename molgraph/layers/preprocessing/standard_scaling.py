import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
from keras.layers.preprocessing import preprocessing_utils as utils

from typing import Tuple
from typing import List
from typing import Optional
from typing import Union
from typing import Tuple


from molgraph.tensors.graph_tensor import GraphTensor


@keras.utils.register_keras_serializable(package='molgraph')
class StandardScaling(layers.experimental.preprocessing.PreprocessingLayer):

    def __init__(
        self,
        mean: Optional[tf.Tensor] = None,
        variance: Optional[tf.Tensor] = None,
        variance_threshold: Optional[Union[float, int]] = None,
        **kwargs
    ):

        if 'feature' in kwargs:
            self.feature = kwargs.pop('feature')
        elif not hasattr(self, 'feature'):
            self.feature = 'node_feature'

        super().__init__(**kwargs)
        self.input_mean = mean
        self.input_variance = variance
        self.variance_threshold = variance_threshold

    def build(self, input_shape):
        super().build(input_shape)

        self.adapt_mean = self.add_weight(
            name='mean',
            shape=input_shape[-1:],
            dtype=tf.float32,
            initializer='zeros',
            trainable=False)

        self.adapt_variance = self.add_weight(
            name='variance',
            shape=input_shape[-1:],
            dtype=tf.float32,
            initializer='ones',
            trainable=False)

        if self.variance_threshold is not None:

            self.adapt_keep = self.add_weight(
                name='keep',
                shape=input_shape[-1:],
                dtype=tf.bool,
                initializer=initializers.Constant(True),
                trainable=False)

        if self.input_mean is None:

            self.count = self.add_weight(
                name='count',
                shape=(),
                dtype=tf.int64,
                initializer='zeros',
                trainable=False)

            self.finalize_state()

        else:
            self.adapt_mean.assign(self.input_mean)
            self.adapt_variance.assign(self.input_variance)

            self.mean = self.adapt_mean
            self.variance = self.adapt_variance

            if self.variance_threshold is not None:
                self.adapt_keep.assign(
                    tf.where(self.variance > self.variance_threshold, True, False))

                self.keep = tf.where(self.adapt_keep == True)[:, 0]

    def adapt(self, data, batch_size=None, steps=None):
        if not isinstance(data,  GraphTensor):
            data = data.map(lambda x: getattr(x, self.feature))
        else:
            data = getattr(data, self.feature)
        super().adapt(data, batch_size=batch_size, steps=steps)

    def update_state(self, feature):

        if self.input_mean is not None:
            raise ValueError("Cannot adapt")

        if isinstance(feature, tf.Tensor):
            batch_count = tf.shape(feature)[0]
            axis = [0]
        else:
            batch_count = tf.shape(feature.to_tensor())[0]
            batch_count += tf.reduce_sum(feature.row_lengths())
            axis = [0, 1]

        batch_count = tf.cast(batch_count, tf.int64)

        total_count = batch_count + self.count

        batch_weight = tf.cast(
            batch_count / total_count, dtype=self._dtype)

        existing_weight = 1. - batch_weight

        # mean
        batch_mean = tf.math.reduce_mean(feature, axis)

        total_mean = (
            self.adapt_mean * existing_weight + batch_mean * batch_weight)

        self.adapt_mean.assign(total_mean)

        # variance
        batch_variance = tf.math.reduce_variance(feature, axis)

        total_variance = (
            (self.adapt_variance + (self.adapt_mean - total_mean)**2) *
            existing_weight +
            (batch_variance + (batch_mean - total_mean)**2) *
            batch_weight)

        self.adapt_variance.assign(total_variance)

        self.count.assign(total_count)

        if self.variance_threshold is not None:
            self.adapt_keep.assign(
                tf.where(self.adapt_variance > self.variance_threshold, True, False))

    def reset_state(self):
        if self.input_mean is not None or not self.built:
            return
        self.adapt_mean.assign(tf.zeros_like(self.adapt_mean))
        self.adapt_variance.assign(tf.ones_like(self.adapt_variance))
        self.count.assign(tf.zeros_like(self.count))
        if self.variance_threshold is not None:
            self.adapt_keep.assign(tf.ones_like(self.adapt_keep))

    def finalize_state(self):
        if self.input_mean is not None or not self.built:
            return
        self.mean = self.adapt_mean
        self.variance = self.adapt_variance
        if self.variance_threshold is not None:
            self.keep = tf.where(self.adapt_keep == True)[:, 0]

    def call(self, data):

        feature = getattr(data, self.feature)

        if isinstance(feature, tf.RaggedTensor):
            gather_axis = 2
            broadcast_shape = (1, 1) + self.mean.shape # unnecessary?
        else:
            gather_axis = 1
            broadcast_shape = (1,) + self.mean.shape # unnecessary?

        mean = tf.reshape(self.mean, broadcast_shape)
        variance = tf.reshape(self.variance, broadcast_shape)

        feature = (
            (feature - mean) /
            tf.maximum(tf.sqrt(variance), keras.backend.epsilon())
        )
        if self.variance_threshold is not None:
            feature = tf.gather(feature, self.keep, axis=gather_axis)

        return data.update({self.feature: feature})

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        config = {
            'mean': utils.listify_tensors(self.mean),
            'variance': utils.listify_tensors(self.variance),
            'feature': self.feature,
            'variance_threshold': self.variance_threshold}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if self.variance_threshold is not None:
            return input_shape[:-1] + (len(self.keep),)
        return input_shape

    def compute_output_signature(self, input_spec):
        if self.variance_threshold is not None:
            shape = input_spec.shape[:-1]
            shape += (len(self.keep),)
            return tf.TensorSpec(shape, dtype=self._dtype)
        return input_spec


@keras.utils.register_keras_serializable(package='molgraph')
class VarianceThreshold(StandardScaling):

    # Solely defined to swap the position of arguments
    def __init__(
        self,
        variance_threshold: Optional[Union[float, int]] = 0.0,
        mean: Optional[tf.Tensor] = None,
        variance: Optional[tf.Tensor] = None,
        **kwargs
    ):
        super().__init__(
            mean=mean,
            variance=variance,
            variance_threshold=variance_threshold,
            **kwargs)

    def call(self, data):

        feature = getattr(data, self.feature)

        if isinstance(feature, tf.Tensor):
            gather_axis = 1
        else:
            gather_axis = 2

        if self.variance_threshold is not None:
            feature = tf.gather(feature, self.keep, axis=gather_axis)

        return data.update({self.feature: feature})


@keras.utils.register_keras_serializable(package='molgraph')
class NodeStandardScaling(StandardScaling):
    feature = 'node_feature'


@keras.utils.register_keras_serializable(package='molgraph')
class EdgeStandardScaling(StandardScaling):
    feature = 'edge_feature'


@keras.utils.register_keras_serializable(package='molgraph')
class NodeVarianceThreshold(VarianceThreshold):
    feature = 'node_feature'


@keras.utils.register_keras_serializable(package='molgraph')
class EdgeVarianceThreshold(VarianceThreshold):
    feature = 'edge_feature'
