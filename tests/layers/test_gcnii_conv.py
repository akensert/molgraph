from tests.layers._common import BaseLayerTestCase

import unittest

import tensorflow as tf

from keras import initializers
from keras import regularizers
from keras import constraints
from keras import layers

from molgraph.layers.convolutional.gcnii_conv import GCNIIConv


PARAMETERS = [
    dict(
        units=8,
        alpha=0.5,
        beta=0.5,
        variant=False,
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
        units=32,
        alpha=1.5,
        beta=1.5,
        variant=True,
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
        alpha=0.0,
        beta=0.5,
        variant=False,
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
        alpha=0.5,
        beta=0.,
        variant=False,
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

class TestGCNIIConv(BaseLayerTestCase):

    def test_output_shape(self):
        self._test_output_shape(GCNIIConv, PARAMETERS)

    def test_output_value(self):
        self._test_output_value(GCNIIConv, PARAMETERS)
   

if __name__ == "__main__":
    unittest.main()
