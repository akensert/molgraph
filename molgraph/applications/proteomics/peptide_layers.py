import tensorflow as tf 
from tensorflow import keras

from molgraph.internal import register_keras_serializable 
from molgraph.tensors.graph_tensor import GraphTensor
from molgraph.applications.proteomics.definitions import _residue_indicator


@register_keras_serializable(package='molgraph')
class _ResidueReadout(keras.layers.Layer):

    def call(self, tensor: GraphTensor) -> tf.Tensor:
        residue_indicator = getattr(tensor, _residue_indicator)
        residue_sizes = tf.math.segment_max(
            residue_indicator, tensor.graph_indicator) + 1
        incr = tf.concat([[0], tf.cumsum(residue_sizes)[:-1]], axis=0)
        residue_indicator += tf.repeat(incr, tensor.sizes)
        residue_feature = tf.math.segment_mean(
            tensor.node_feature, residue_indicator)
        return tf.RaggedTensor.from_row_lengths(
            residue_feature, residue_sizes).to_tensor()