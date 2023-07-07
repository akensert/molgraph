from tests.layers._common import BaseGeometricLayerTestCase

import unittest

import tensorflow as tf

from keras import initializers
from keras import regularizers
from keras import constraints
from keras import layers

from molgraph.layers.geometric.gcf_conv import GCFConv


PARAMETERS = [
    dict(
        units=8,
        distance_min=-1.0,
        distance_max=18.0,
        distance_granularity=0.1,
        rbf_stddev='auto',
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
        distance_min=-1.0,
        distance_max=30.0,
        distance_granularity=0.5,
        rbf_stddev=0.1,
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
        distance_min=-1.0,
        distance_max=18.0,
        distance_granularity=0.01,
        rbf_stddev='auto',
        self_projection=True,

        normalization='layer_norm',
        residual=False,
        dropout=None,
        activation=tf.keras.layers.ReLU(),
        use_bias=False,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None
    ),
    dict(
        units=33,
        distance_min=-10.0,
        distance_max=30.0,
        distance_granularity=0.05,
        rbf_stddev=0.5,
        self_projection=False,

        normalization='batch_norm',
        residual=False,
        dropout=None,
        activation=tf.keras.layers.ReLU(),
        use_bias=False,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None
    ),
]

class TestGCFConv(BaseGeometricLayerTestCase):

    def test_output_shape(self):
        self._test_output_shape(GCFConv, PARAMETERS)

    def test_output_value(self):
        self._test_output_value(GCFConv, PARAMETERS)
   

if __name__ == "__main__":
    unittest.main()
