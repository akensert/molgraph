import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.keras import activations

from keras.utils import tf_utils


@keras.utils.register_keras_serializable(package='molgraph')
class FeatureProjection(layers.Layer):

    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):

        if 'feature' in kwargs:
            self.feature = kwargs.pop('feature')
        elif not hasattr(self, 'feature'):
            self.feature = 'node_feature'

        super().__init__(
            activity_regularizer=activity_regularizer, **kwargs)

        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.projection = layers.Dense(
            units=self.units,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint)

    def call(self, tensor):
        tensor_orig = tensor
        if isinstance(getattr(tensor, self.feature), tf.RaggedTensor):
            tensor = tensor.merge()
        return tensor_orig.update({
            self.feature: self.projection(getattr(tensor, self.feature))})

    def get_config(self):
        config = {
            'feature': self.feature,
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@keras.utils.register_keras_serializable(package='molgraph')
class NodeFeatureProjection(FeatureProjection):
    feature = 'node_feature'


@keras.utils.register_keras_serializable(package='molgraph')
class EdgeFeatureProjection(FeatureProjection):
    feature = 'edge_feature'
