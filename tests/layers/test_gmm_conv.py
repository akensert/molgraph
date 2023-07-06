from tests.layers._common import BaseLayerTestCase

import unittest

import tensorflow as tf
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import activations
from keras import layers

from molgraph.layers.attentional.gmm_conv import GMMConv


PARAMETERS = [
    dict(
        units=8,
        num_kernels=2,
        merge_mode='concat',
        pseudo_coord_dim=10,
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
        num_kernels=8,
        merge_mode='sum',
        pseudo_coord_dim=1,
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
        num_kernels=8,
        merge_mode='mean',
        pseudo_coord_dim=5,
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
        num_kernels=5,
        merge_mode='sum',
        pseudo_coord_dim=2,
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

class TestGMMConv(BaseLayerTestCase):

    def test_output_shape(self):
        self._test_output_shape(GMMConv, PARAMETERS)

    def test_output_value(self):
        self._test_output_value(GMMConv, PARAMETERS)
   

if __name__ == "__main__":
    unittest.main()
