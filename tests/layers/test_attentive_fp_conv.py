from tests.layers._common import BaseLayerTestCase

import unittest

import tensorflow as tf

from keras import initializers
from keras import regularizers
from keras import constraints
from keras import layers

from molgraph.layers.attentional.attentive_fp_conv import AttentiveFPConv


PARAMETERS = [
    dict(
        units=8,
        apply_initial_node_projection=True,
        gru_cell=None,
        num_heads=2,
        merge_mode='concat',
        self_projection=True,

        normalization=True,
        residual=True,
        dropout=0.5,
        activation='relu',
        use_bias=True,
        kernel_initializer=initializers.GlorotUniform(),
        bias_initializer=initializers.GlorotUniform(),
        kernel_regularizer=regularizers.L1L2(),
        bias_regularizer=regularizers.L1L2(),
        activity_regularizer=regularizers.L1L2(),
        kernel_constraint=constraints.MinMaxNorm(),
        bias_constraint=constraints.MinMaxNorm()
    ),
    dict(
        units=32,
        apply_initial_node_projection=True,
        gru_cell=layers.GRUCell(32),
        num_heads=8,
        merge_mode='concat',
        self_projection=False,
        
        normalization=False,
        residual=True,
        dropout=None,
        activation=layers.ReLU(),
        use_bias=False,
        kernel_initializer=initializers.GlorotUniform(),
        bias_initializer=initializers.GlorotUniform(),
        kernel_regularizer=regularizers.L1L2(),
        bias_regularizer=regularizers.L1L2(),
        activity_regularizer=regularizers.L1L2(),
        kernel_constraint=constraints.MinMaxNorm(),
        bias_constraint=constraints.MinMaxNorm()
    ),
    dict(
        units=None,
        apply_initial_node_projection=False,
        gru_cell=None,
        num_heads=8,
        merge_mode='mean',
        self_projection=True,

        normalization='layer_norm',
        residual=False,
        dropout=None,
        activation=tf.keras.layers.ReLU(),
        use_bias=False,
        kernel_initializer=None,
        bias_initializer=None,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None
    ),
    dict(
        units=33,
        apply_initial_node_projection=False,
        gru_cell=layers.GRUCell(33),
        num_heads=5,
        merge_mode='mean',
        self_projection=False,

        normalization='batch_norm',
        residual=False,
        dropout=None,
        activation=tf.keras.layers.ReLU(),
        use_bias=False,
        kernel_initializer=None,
        bias_initializer=None,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None
    ),
]

class TestAttentiveFPConv(BaseLayerTestCase):

    def test_output_shape(self):
        self._test_output_shape(AttentiveFPConv, PARAMETERS)

    def test_output_value(self):
        self._test_output_value(AttentiveFPConv, PARAMETERS)
   

if __name__ == "__main__":
    unittest.main()
