from tests.layers._common import BaseLayerTestCase

import unittest

import tensorflow as tf

from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.keras import layers

from molgraph.layers.convolutional.graph_sage_conv import GraphSageConv


PARAMETERS = [
    dict(
        units=8,
        aggregation_mode='max',
        normalize=True,
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
        aggregation_mode='lstm',
        normalize=True,
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
        aggregation_mode='max',
        normalize=False,
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
        aggregation_mode='mean',
        normalize=True,
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

class TestGraphSageConv(BaseLayerTestCase):

    def test_output_shape(self):
        self._test_output_shape(GraphSageConv, PARAMETERS)

    def test_output_value(self):
        self._test_output_value(GraphSageConv, PARAMETERS)
   

if __name__ == "__main__":
    unittest.main()
