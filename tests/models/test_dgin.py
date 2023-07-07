import molgraph
import unittest

import tensorflow as tf

from molgraph.models import DGIN
from molgraph import layers

from tests.models._common import graph_tensor
from tests.models._common import graph_tensor_merged


class TestDMPNN(unittest.TestCase):

    def test_model_with_ragged_tensor(self):
        inputs = tf.keras.layers.Input(type_spec=graph_tensor.unspecific_spec)
        x = DGIN(units=32, name='dgin')(inputs)
        x = layers.SetGatherReadout(name='readout')(x)
        outputs = tf.keras.layers.Dense(10, activation='sigmoid')(x)
        mpnn_classifier = tf.keras.Model(inputs, outputs)
        preds = mpnn_classifier.predict(graph_tensor, verbose=0)
        assert preds.shape == (3, 10)

    def test_model_with_nonragged_tensor(self):
        inputs = tf.keras.layers.Input(
            type_spec=graph_tensor_merged.unspecific_spec)
        x = DGIN(units=32, name='dgin')(inputs)
        x = layers.SetGatherReadout(name='readout')(x)
        outputs = tf.keras.layers.Dense(10, activation='sigmoid')(x)
        mpnn_classifier = tf.keras.Model(inputs, outputs)
        preds = mpnn_classifier.predict(graph_tensor, verbose=0)
        assert preds.shape == (3, 10)


if __name__ == "__main__":
    unittest.main()