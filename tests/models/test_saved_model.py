import molgraph
import unittest

import tensorflow as tf

import tempfile
import shutil
import numpy as np

from molgraph import layers as gnn_layers

from tests.models._common import graph_tensor
from tests.models._common import graph_tensor_merged


class TestSavedModelAPI(unittest.TestCase):

    def test_saved_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(type_spec=graph_tensor.unspecific_spec),
            gnn_layers.attentional.gt_conv.GTConv(128),
            gnn_layers.attentional.gt_conv.GTConv(128),
            gnn_layers.readout.segment_pool.SegmentPoolingReadout(),
            tf.keras.layers.Dense(1),
        ])
        output_before = model(graph_tensor)
        file = tempfile.NamedTemporaryFile()
        filename = file.name
        file.close()
        tf.saved_model.save(model, filename)
        loaded_model = tf.saved_model.load(filename)
        output_after = loaded_model(graph_tensor)
        shutil.rmtree(filename)

        test = np.all(output_before.numpy().round(5) == output_after.numpy().round(5))
        self.assertTrue(test)

    def test_saved_model_keras(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(type_spec=graph_tensor.unspecific_spec),
            gnn_layers.attentional.gt_conv.GTConv(128),
            gnn_layers.attentional.gt_conv.GTConv(128),
            gnn_layers.readout.segment_pool.SegmentPoolingReadout(),
            tf.keras.layers.Dense(1),
        ])
        _ = model(graph_tensor)
        weights_before = model.trainable_weights[0]
        file = tempfile.NamedTemporaryFile()
        filename = file.name
        file.close()
        model.save(filename)
        loaded_model = tf.keras.models.load_model(filename)
        weights_after = loaded_model.trainable_weights[0]
        
        shutil.rmtree(filename)
        test = tf.reduce_all(weights_before == weights_after).numpy()
        self.assertTrue(test)

    def test_saved_model_merged_graph_tensor(self):
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(type_spec=graph_tensor_merged.unspecific_spec),
            gnn_layers.attentional.gt_conv.GTConv(128),
            gnn_layers.attentional.gt_conv.GTConv(128),
            gnn_layers.readout.segment_pool.SegmentPoolingReadout(),
            tf.keras.layers.Dense(1),
        ])
        output_before = model(graph_tensor_merged)
        file = tempfile.NamedTemporaryFile()
        filename = file.name
        file.close()
        tf.saved_model.save(model, filename)
        loaded_model = tf.saved_model.load(filename)
        output_after = loaded_model(graph_tensor_merged)

        shutil.rmtree(filename)

        test = np.all(output_before.numpy().round(5) == output_after.numpy().round(5))
        self.assertTrue(test)


if __name__ == "__main__":
    unittest.main()