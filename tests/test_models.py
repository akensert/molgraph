import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import tempfile
import os
import shutil
import numpy as np

from molgraph import layers
from molgraph.chemistry.molecular_encoders import MolecularGraphEncoder
from molgraph.chemistry.atomic.featurizers import AtomicFeaturizer
from molgraph.chemistry.atomic import features

from molgraph import layers as gnn_layers


import pytest

# Define atomic encoders
atom_encoder = AtomicFeaturizer([
    features.Symbol({'C', 'N', 'O'}, oov_size=1),
    features.Hybridization({'SP', 'SP2', 'SP3'}, oov_size=1),
    features.HydrogenDonor(),
    features.HydrogenAcceptor(),
    features.Hetero()
])
bond_encoder = AtomicFeaturizer([
    features.BondType({'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'}),
    features.Rotatable()
])
# Define molecular graph encoder
encoder = MolecularGraphEncoder(atom_encoder, bond_encoder)

# Typical graph, with nested ragged tensors
graph_tensor = encoder([
    'OCC1OC(C(C1O)O)n1cnc2c1ncnc2N', 'C(C(=O)O)N', '[Na+].[O-]c1ccccc1'])


@pytest.mark.parametrize('graph_tensor', [graph_tensor])
def test_saved_model(graph_tensor) -> None:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(type_spec=graph_tensor.unspecific_spec),
        gnn_layers.attentional.gat_conv.GATConv(128),
        gnn_layers.attentional.gat_conv.GATConv(128),
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

    assert np.all(output_before.numpy().round(5) == output_after.numpy().round(5))

    shutil.rmtree(filename)


@pytest.mark.parametrize('graph_tensor', [graph_tensor])
def test_saved_model_keras(graph_tensor) -> None:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(type_spec=graph_tensor.unspecific_spec),
        gnn_layers.attentional.gat_conv.GATConv(128),
        gnn_layers.attentional.gat_conv.GATConv(128),
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
    assert tf.reduce_all(weights_before == weights_after).numpy()
    shutil.rmtree(filename)


@pytest.mark.parametrize('graph_tensor', [graph_tensor])
def test_saved_model_merged_graph_tensor(graph_tensor) -> None:
    graph_tensor = graph_tensor.merge()
    model = tf.keras.Sequential([
        tf.keras.layers.Input(type_spec=graph_tensor.unspecific_spec),
        gnn_layers.attentional.gat_conv.GATConv(128),
        gnn_layers.attentional.gat_conv.GATConv(128),
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

    assert np.all(output_before.numpy().round(5) == output_after.numpy().round(5))

    shutil.rmtree(filename)
