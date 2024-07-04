import molgraph
import unittest

import tensorflow as tf

from molgraph.models import MPNN
from molgraph import layers

from tests.models._common import graph_tensor
from tests.models._common import graph_tensor_merged



class TestMPNN(unittest.TestCase):

    def test_model_with_ragged_tensor(self):
        inputs = layers.GNNInput(type_spec=graph_tensor.spec)
        x = MPNN(units=32, steps=4, name='mpnn')(inputs)
        x = layers.SetGatherReadout(name='readout')(x)
        outputs = tf.keras.layers.Dense(10, activation='sigmoid')(x)
        mpnn_classifier = tf.keras.Model(inputs, outputs)
        preds = mpnn_classifier.predict(graph_tensor, verbose=0)
        assert preds.shape == (3, 10)

    def test_model_with_nonragged_tensor(self):
        inputs = layers.GNNInput(
            type_spec=graph_tensor_merged.spec)
        x = MPNN(units=32, steps=4, name='mpnn')(inputs)
        x = layers.SetGatherReadout(name='readout')(x)
        outputs = tf.keras.layers.Dense(10, activation='sigmoid')(x)
        mpnn_classifier = tf.keras.Model(inputs, outputs)
        preds = mpnn_classifier.predict(graph_tensor, verbose=0)
        assert preds.shape == (3, 10)


if __name__ == "__main__":
    unittest.main()