import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import numpy as np

from molgraph import layers
from molgraph.chemistry import MolecularGraphEncoder
from molgraph.chemistry import Featurizer
from molgraph.chemistry import features
from molgraph.chemistry.ops import molecule_from_string

import pytest


def test_atomic_encoder():

    rdkit_mol = molecule_from_string('OCC1OC(C(C1O)O)n1cnc2c1ncnc2N')
    atom = rdkit_mol.GetAtoms()[0]
    bond = rdkit_mol.GetBonds()[0]

    atom_encoder = Featurizer([features.Symbol({'C', 'N'}, oov_size=0)])
    assert np.all(atom_encoder(atom) == np.array([0., 0.]))

    atom_encoder = Featurizer([features.Symbol({'C', 'N'}, oov_size=1)])
    assert np.all(atom_encoder(atom) == np.array([1., 0., 0.]))

    atom_encoder = Featurizer([features.Symbol({'C', 'N', 'O'})])
    assert np.all(atom_encoder(atom) == np.array([0., 0., 1.]))

    atom_encoder = Featurizer([
        features.Symbol({'C', 'N', 'O'}),
        features.Hybridization({'SP', 'SP2', 'SP3'}),
        features.HydrogenDonor(),
        features.HydrogenAcceptor(),
        features.Hetero()
    ])
    assert np.all(atom_encoder(atom) ==
                  np.array([0., 0., 1., 0., 0., 1., 1., 1., 1.]))

    atom_encoder = Featurizer([
        features.Symbol(['C', 'N', 'O', 'P'], ordinal=True),
        features.Hybridization(['SP', 'SP2', 'SP3', 'SP3D'], ordinal=True),
        features.HydrogenDonor(),
        features.HydrogenAcceptor(),
        features.Hetero()
    ])

    assert np.all(atom_encoder(atom) ==
                  np.array([1., 1., 1., 0., 1., 1., 1., 0., 1., 1., 1.]))


def test_molecular_encoders():

    node_feature = tf.constant(
        [[1., 0., 0., 0., 0., 1., 0., 0., 0.],
         [1., 0., 0., 0., 1., 0., 0., 0., 0.],
         [0., 0., 1., 0., 1., 0., 0., 1., 1.],
         [0., 0., 1., 0., 1., 0., 1., 0., 1.],
         [0., 1., 0., 0., 0., 1., 1., 1., 1.]], dtype=tf.float32)
    edge_feature = tf.constant(
        [[0., 0., 1., 0., 1.],
         [0., 0., 1., 0., 0.],
         [0., 0., 1., 0., 1.],
         [0., 1., 0., 0., 0.],
         [0., 0., 1., 0., 0.],
         [0., 1., 0., 0., 0.],
         [0., 0., 1., 0., 0.],
         [0., 0., 1., 0., 0.]], dtype=tf.float32)
    edge_src = tf.constant([0, 0, 1, 1, 1, 2, 3, 4], dtype=tf.int32)
    edge_dst = tf.constant([1, 4, 0, 2, 3, 1, 1, 0], dtype=tf.int32)

    # Define atomic encoders
    atom_encoder = Featurizer([
        features.Symbol({'C', 'N', 'O'}),
        features.Hybridization({'SP', 'SP2', 'SP3'}),
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
        'OCC1OC(C(C1O)O)n1cnc2c1ncnc2N',
        'C(C(=O)O)N',
        'C1=CC(=CC=C1CC(C(=O)O)N)O'
    ])

    assert tf.reduce_all(graph_tensor[1].node_feature == node_feature)
    assert tf.reduce_all(graph_tensor[1].edge_feature == edge_feature)
    assert tf.reduce_all(graph_tensor[1].edge_src == edge_src)
    assert tf.reduce_all(graph_tensor[1].edge_dst == edge_dst)

    assert tf.reduce_all(graph_tensor.merge()[1].node_feature == node_feature)
    assert tf.reduce_all(graph_tensor.merge()[1].edge_feature == edge_feature)
    assert tf.reduce_all(graph_tensor.merge()[1].edge_src == edge_src)
    assert tf.reduce_all(graph_tensor.merge()[1].edge_dst == edge_dst)
    
