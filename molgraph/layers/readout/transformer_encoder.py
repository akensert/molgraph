import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import activations

from typing import Union
from typing import Callable

from molgraph.internal import register_keras_serializable 

from molgraph.tensors.graph_tensor import GraphTensor


@register_keras_serializable(package='molgraph')
class TransformerEncoderReadout(layers.Layer):

    '''Transformer encoder layer for graph readout.

    Example usage:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     sizes=[2, 3],
    ...     node_feature=[[1., 0.], [1., 0.], [1., 0.], [1., 0.], [0., 1.]],
    ...     edge_src=[1, 0, 3, 4, 2, 4, 3, 2],
    ...     edge_dst=[0, 1, 2, 2, 3, 3, 4, 4],
    ... )
    >>> model = tf.keras.Sequential([
    ...     # molgraph.layers.GCNConv(4),
    ...     molgraph.layers.TransformerEncoderReadout()
    ... ])
    >>> model(graph_tensor).shape
    TensorShape([2, 2])

    Args:
        hidden_units (int):
            Number of hidden units (of the feedforward network).
        num_heads (int):
            Number of attention heads. Default to 8.
        activation (str, tf.keras.activations.Activation, None):
            The activation function applied to the output. Default to 'relu'.
    '''

    def __init__(
        self,
        hidden_units: int = 256,
        num_heads: int = 8,
        activation: Union[
            None, str, Callable[[tf.Tensor], tf.Tensor]] = 'relu',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.activation = activations.get(activation)
        self.layernorm_1 = keras.layers.LayerNormalization()
        self.layernorm_2 = keras.layers.LayerNormalization()
        self.average_pooling = keras.layers.GlobalAveragePooling1D()

        self._built = False
        self._node_feature_shape = None

    def _build(
        self,
        node_feature: Union[tf.Tensor, tf.TensorShape]
    ) -> None:
        '''Custom build method for initializing additional attributes.

        Args:
            node_feature (tf.Tensor, tf.TensorShape):
                Either the shape of the ``node_feature`` field of
                GraphTensor, or the node_feature field itself.
        '''

        self._built = True

        if hasattr(node_feature, "shape"):
            self._node_feature_shape = tf.TensorShape(node_feature.shape)
        else:
            self._node_feature_shape = tf.TensorShape(node_feature)

        node_dim = self._node_feature_shape[-1]

        with tf.init_scope():
            self.attention = keras.layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=node_dim)
            self.attention._build_from_signature(
                self._node_feature_shape, self._node_feature_shape)
            self.projection = keras.Sequential([
                keras.layers.Dense(self.hidden_units, self.activation),
                keras.layers.Dense(node_dim)])

    def call(self, tensor: GraphTensor) -> tf.Tensor:
        '''Defines the computation from inputs to outputs.

        This method should not be called directly, but indirectly
        via ``__call__()``. Upon first call, the layer is automatically
        built via ``_build()``.

        Args:
            tensor (GraphTensor):
                Input to the layer.

        Returns:
            A ``tf.Tensor`` or `tf.RaggedTensor` based on the node_feature
            field of the inputted ``GraphTensor``.
        '''
        node_feature = tensor.node_feature

        if isinstance(node_feature, tf.Tensor):
            graph_indicator = tensor.graph_indicator
            node_feature = tf.RaggedTensor.from_value_rowids(
                node_feature, graph_indicator)

        x = node_feature.to_tensor()

        if not self._built:
            self._build(x)

        # Compute padding mask for attention layer
        padding_mask = tf.reduce_any(tf.not_equal(x, 0.0), axis=-1)
        padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]

        attention_output = self.attention(x, x, attention_mask=padding_mask)

        proj_input = self.layernorm_1(x + attention_output)
        proj_output = self.layernorm_2(proj_input + self.projection(proj_input))
        return self.average_pooling(proj_output)

    def compute_output_shape(self, input_shape):
        if input_shape[0] is None and input_shape[1] is not None:
            # input_shape corresponds to a tf.Tensor
            return input_shape
        # input_shape corresponds to a tf.RaggedTensor
        return input_shape[1:]

    def get_config(self):
        config = super().get_config()
        config.update({
            'hidden_units': self.hidden_units,
            'num_heads': self.num_heads,
            'activation': activations.serialize(self.activation),
            'node_feature_shape': self._node_feature_shape,
        })
        return config

    @classmethod
    def from_config(cls, config):
        node_feature_shape = config.pop("node_feature_shape")
        layer = cls(**config)
        if node_feature_shape is None:
            pass
        else:
            layer._build(node_feature_shape)
        return layer
