from tests.layers._common import BaseLayerTestCase

import unittest

import tensorflow as tf
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import activations
from keras import layers

from molgraph.layers.message_passing.edge_conv import EdgeConv
from molgraph.layers.readout.node_readout import NodeReadout

from tests.layers._common import inputs


PARAMETERS = [
    dict(
        units=8,
        update_fn=None,
        update_mode='gru',

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
        update_fn=layers.Dense(32),
        update_mode='dense',

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
        update_fn=None,
        update_mode='gru',

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
        update_fn=layers.GRUCell(33),
        update_mode='dense',

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

class TestEdgeConv(unittest.TestCase):
    pass

   

if __name__ == "__main__":
    unittest.main()
