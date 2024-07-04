import molgraph
import unittest

import tensorflow as tf

import tempfile
import shutil
import numpy as np
import os

from molgraph import layers as gnn_layers

from tests.models._common import graph_tensor
from tests.models._common import graph_tensor_merged

# TODO: Add tests for Keras Functional API

class TestSavedModelAPI(unittest.TestCase):

    def test_saved_model_keras_1(self):
        model = tf.keras.Sequential([
            gnn_layers.GNNInputLayer(type_spec=graph_tensor.spec),
            gnn_layers.positional_encoding.laplacian.LaplacianPositionalEncoding(),
            gnn_layers.attentional.gt_conv.GTConv(128),
            gnn_layers.attentional.gt_conv.GTConv(128),
            gnn_layers.readout.segment_pool.SegmentPoolingReadout(),
            tf.keras.layers.Dense(1),
        ])
        output_before = model(graph_tensor)
        file = tempfile.NamedTemporaryFile()
        filename = file.name
        file.close()
        tf.keras.models.save_model(model, filename + '.keras')
        loaded_model = tf.keras.models.load_model(filename + '.keras')
        output_after = loaded_model(graph_tensor)
        os.remove(filename + '.keras')

        test = np.all(output_before.numpy().round(5) == output_after.numpy().round(5))
        self.assertTrue(test)

    def test_saved_model_keras_2(self):
        model = tf.keras.Sequential([
            gnn_layers.GNNInputLayer(type_spec=graph_tensor.spec),
            gnn_layers.positional_encoding.laplacian.LaplacianPositionalEncoding(),
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
        model.save(filename + '.keras')
        loaded_model = tf.keras.models.load_model(filename + '.keras')
        weights_after = loaded_model.trainable_weights[0]
        
        os.remove(filename + '.keras')
        test = tf.reduce_all(weights_before == weights_after).numpy()
        self.assertTrue(test)

    def test_saved_model(self):
        model = tf.keras.Sequential([
            gnn_layers.GNNInputLayer(type_spec=graph_tensor.spec),
            gnn_layers.positional_encoding.laplacian.LaplacianPositionalEncoding(),
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

    def test_saved_model_merged_graph_tensor(self):
        
        model = tf.keras.Sequential([
            gnn_layers.GNNInputLayer(type_spec=graph_tensor_merged.spec),
            gnn_layers.positional_encoding.laplacian.LaplacianPositionalEncoding(),
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

    def test_saved_model_merged_graph_tensor_no_excplit_spec_and_different_size(self):
        
        model = tf.keras.Sequential([
            # gnn_layers.GNNInputLayer(type_spec=graph_tensor_merged.spec),
            gnn_layers.positional_encoding.laplacian.LaplacianPositionalEncoding(),
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
        output_after = loaded_model(graph_tensor_merged[:1])

        shutil.rmtree(filename)

        test = np.all(output_before[:1].numpy().round(5) == output_after.numpy().round(5))
        self.assertTrue(test)


if __name__ == "__main__":
    unittest.main()