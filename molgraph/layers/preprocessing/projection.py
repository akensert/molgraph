import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import activations

from molgraph.internal import register_keras_serializable 


@register_keras_serializable(package='molgraph')
class FeatureProjection(layers.Layer):

    '''Feature projection via dense layer.

    Specify, as keyword argument only,
    ``FeatureProjection(feature='node_feature')`` to perform a projection
    of the ``node_feature`` component of the ``GraphTensor``, or,
    ``FeatureProjection(feature='edge_feature')`` to perform a projection
    of the ``edge_feature`` component of the ``GraphTensor``. If not specified,
    the ``node_feature`` component will be considered.

    Instead of specifying `feature`, ``NodeFeatureProjection(...)`` or 
    ``EdgeFeatureProjection(...)`` can be used instead.

    Example usage:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     sizes=[2, 3],
    ...     node_feature=[[1., 0.], [1., 0.], [1., 0.], [1., 0.], [0., 1.]],
    ...     edge_src=[1, 0, 3, 4, 2, 4, 3, 2],
    ...     edge_dst=[0, 1, 2, 2, 3, 3, 4, 4],
    ... )
    >>> model = tf.keras.Sequential([
    ...     molgraph.layers.FeatureProjection(
    ...         feature='node_feature', units=16)
    ... ])
    >>> model(graph_tensor).node_feature.shape
    TensorShape([5, 16])

    Args:
        units (int, None):
            Number of output units.
        activation (tf.keras.activations.Activation, callable, str, None):
            Activation function applied to the output of the layer. Default to None.
        use_bias (bool):
            Whether the layer should use biases. Default to False.
        kernel_initializer (tf.keras.initializers.Initializer, str):
            Initializer function for the kernel. Default to
            tf.keras.initializers.TruncatedNormal(stddev=0.005).
        bias_initializer (tf.keras.initializers.Initializer, str):
            Initializer function for the bias. Default to
            tf.keras.initializers.Constant(0.).
        kernel_regularizer (tf.keras.regularizers.Regularizer, None):
            Regularizer function applied to the kernel. Default to None.
        bias_regularizer (tf.keras.regularizers.Regularizer, None):
            Regularizer function applied to the bias. Default to None.
        activity_regularizer (tf.keras.regularizers.Regularizer, None):
            Regularizer function applied to the final output of the layer.
            Default to None.
        kernel_constraint (tf.keras.constraints.Constraint, None):
            Constraint function applied to the kernel. Default to None.
        bias_constraint (tf.keras.constraints.Constraint, None):
            Constraint function applied to the bias. Default to None.
        **kwargs:
            Specify the relevant ``feature``. Default to ``node_feature``.
            The reminaing kwargs are passed to the parent class.
    '''

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
        '''Defines the computation from inputs to outputs.

        This method should not be called directly, but indirectly
        via ``__call__()``. Upon first call, the layer is automatically
        built via ``build()``.

        Args:
            tensor (GraphTensor):
                Input to the layer.

        Returns:
            A ``tf.Tensor`` or `tf.RaggedTensor` based on the ``node_feature``
            component of the inputted ``GraphTensor``.
        '''
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


@register_keras_serializable(package='molgraph')
class NodeFeatureProjection(FeatureProjection):
    feature = 'node_feature'


@register_keras_serializable(package='molgraph')
class EdgeFeatureProjection(FeatureProjection):
    feature = 'edge_feature'
