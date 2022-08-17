import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
from keras.utils import tf_utils

from typing import Tuple
from typing import Union
from typing import Callable

from molgraph.tensors.graph_tensor import GraphTensor



@keras.utils.register_keras_serializable(package='molgraph')
class TransformerEncoderReadout(layers.Layer):

    'Transformer encoder layer for graph readout.'

    def __init__(
        self,
        units: int = 256,
        num_heads: int = 8,
        activation: Union[
            None, str, Callable[[tf.Tensor], tf.Tensor]] = 'relu',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.activation = activations.get(activation)
        self.layernorm_1 = keras.layers.LayerNormalization()
        self.layernorm_2 = keras.layers.LayerNormalization()
        self.average_pooling = keras.layers.GlobalAveragePooling1D()

        self._built_from_signature = False
        self._node_feature_shape = None

    def _build_from_signature(
        self,
        node_feature: Union[tf.Tensor, tf.TensorShape]
    ) -> None:
        '''Custom build method for initializing additional attributes.

        Args:
            node_feature (tf.Tensor, tf.TensorShape):
                Either the shape of the node_feature component of GraphTensor,
                or the node_feature component itself.
        '''

        self._built_from_signature = True

        if hasattr(node_feature, "shape"):
            self._node_feature_shape = tf.TensorShape(node_feature.shape)
        else:
            self._node_feature_shape = tf.TensorShape(node_feature)

        node_dim = self._node_feature_shape[-1]

        with tf_utils.maybe_init_scope(self):
            self.attention = keras.layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=node_dim)
            self.attention._build_from_signature(
                self._node_feature_shape, self._node_feature_shape)
            self.projection = keras.Sequential([
                keras.layers.Dense(self.units, self.activation),
                keras.layers.Dense(node_dim)])

    def call(self, tensor: GraphTensor) -> tf.Tensor:
        '''Defines the computation from inputs to outputs.

        This method should not be called directly, but indirectly
        via ``__call__()``. Upon first call, the layer is automatically
        built via ``_build_from_signature()``.

        Args:
            tensor (GraphTensor):
                A graph tensor which serves as input to the layer.

        Returns:
            tf.Tensor:
                A tensor based on the node_feature component of the inputted
                graph tensor.
        '''
        node_feature = tensor.node_feature

        if isinstance(node_feature, tf.Tensor):
            graph_indicator = tensor.graph_indicator
            node_feature = tf.RaggedTensor.from_value_rowids(
                node_feature, graph_indicator)

        x = node_feature.to_tensor()

        if not self._built_from_signature:
            self._build_from_signature(x)

        # Compute padding mask for attention layer
        padding_mask = tf.reduce_any(tf.not_equal(x, 0.0), axis=-1)
        padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]

        attention_output = self.attention(x, x, attention_mask=padding_mask)

        proj_input = self.layernorm_1(x + attention_output)
        proj_output = self.layernorm_2(proj_input + self.projection(proj_input))
        return self.average_pooling(proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
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
            pass # TODO(akensert): add warning message about not restoring weights
        else:
            layer._build_from_signature(node_feature_shape)
        return layer
