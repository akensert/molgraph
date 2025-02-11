from tests.layers._common import BaseLayerTestCase

import unittest

import tensorflow as tf

from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.keras import layers

from molgraph.layers.convolutional.gcn_conv import GCNConv


PARAMETERS = [
    dict(
        units=8,
        num_bases=16,
        degree_normalization='symmetric',
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
        units=8,
        num_bases=3,
        degree_normalization='symmetric',
        self_projection=True,
        use_edge_features=True,
        
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
        num_bases=None,
        degree_normalization=None,
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
        num_bases=3,
        degree_normalization='symmetric',
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
        num_bases=1,
        degree_normalization='row',
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

class TestGCNConv(BaseLayerTestCase):

    def test_output_shape(self):
        self._test_output_shape(GCNConv, PARAMETERS)

    def test_output_value(self):
        self._test_output_value(GCNConv, PARAMETERS)
   

if __name__ == "__main__":
    unittest.main()
