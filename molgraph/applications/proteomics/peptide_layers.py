import tensorflow as tf 
from tensorflow import keras

from molgraph.internal import register_keras_serializable 
from molgraph.tensors.graph_tensor import GraphTensor
from molgraph.applications.proteomics.definitions import _residue_indicator


@register_keras_serializable(package='molgraph')
class _ResidueReadout(keras.layers.Layer):

    def call(self, tensor: GraphTensor) -> tf.Tensor:
        residue_feature = tf.math.segment_mean(
            tensor.node_feature, getattr(tensor, _residue_indicator))
        # Compute row_lengths (number of residues per peptide)
        row_lengths = tf.math.segment_max(
            getattr(tensor, _residue_indicator), tensor.graph_indicator)
        row_lengths = tf.concat([[-1], row_lengths], axis=0)
        row_lengths = row_lengths[1:] - row_lengths[:-1]
        # Obtain a padded "rectangular" tensor for e.g. RNN
        return tf.RaggedTensor.from_row_lengths(
            residue_feature, row_lengths).to_tensor()