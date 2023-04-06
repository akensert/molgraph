import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import tempfile
import os
import shutil
import numpy as np

from molgraph.chemistry import MolecularGraphEncoder
from molgraph.chemistry import Featurizer
from molgraph.chemistry import features
from molgraph.models import MPNN, DMPNN, DGIN
from molgraph import layers

import pytest

# Define atomic encoders
atom_encoder = Featurizer([
    features.Symbol({'C', 'N', 'O'}, oov_size=1),
    features.Hybridization({'SP', 'SP2', 'SP3'}, oov_size=1),
    features.HydrogenDonor(),
    features.HydrogenAcceptor(),
    features.Hetero()
])
bond_encoder = Featurizer([
    features.BondType({'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'}),
    features.Rotatable()
])
# Define molecular graph encoder
encoder = MolecularGraphEncoder(atom_encoder, bond_encoder)

# Typical graph, with nested ragged tensors
graph_tensor = encoder([
    'OCC1OC(C(C1O)O)n1cnc2c1ncnc2N', 'C(C(=O)O)N', '[Na+].[O-]c1ccccc1'])


@pytest.mark.parametrize('graph_tensor', [graph_tensor])
def test_mpnn_model(graph_tensor) -> None:
    inputs = tf.keras.layers.Input(type_spec=graph_tensor.unspecific_spec)
    x = MPNN(units=32, steps=4, name='mpnn')(inputs)
    x = layers.SetGatherReadout(name='readout')(x)
    outputs = tf.keras.layers.Dense(10, activation='sigmoid')(x)
    mpnn_classifier = tf.keras.Model(inputs, outputs)
    # Make predictions
    preds = mpnn_classifier.predict(graph_tensor)
    assert preds.shape == (3, 10)

@pytest.mark.parametrize('graph_tensor', [graph_tensor])
def test_dmpnn_model(graph_tensor) -> None:
    inputs = tf.keras.layers.Input(type_spec=graph_tensor.unspecific_spec)
    x = DMPNN(units=32, steps=4, name='dmpnn')(inputs)
    x = layers.SetGatherReadout(name='readout')(x)
    outputs = tf.keras.layers.Dense(10, activation='sigmoid')(x)
    mpnn_classifier = tf.keras.Model(inputs, outputs)
    # Make predictions
    preds = mpnn_classifier.predict(graph_tensor)
    assert preds.shape == (3, 10)

@pytest.mark.parametrize('graph_tensor', [graph_tensor])
def test_dgin_model(graph_tensor) -> None:
    inputs = tf.keras.layers.Input(type_spec=graph_tensor.unspecific_spec)
    x = DGIN(units=32, name='dgin')(inputs)
    x = layers.SetGatherReadout(name='readout')(x)
    outputs = tf.keras.layers.Dense(10, activation='sigmoid')(x)
    mpnn_classifier = tf.keras.Model(inputs, outputs)
    # Make predictions
    preds = mpnn_classifier.predict(graph_tensor)
    assert preds.shape == (3, 10)

