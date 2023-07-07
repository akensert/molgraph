import molgraph

import tensorflow as tf

import unittest

from molgraph.chemistry import MolecularGraphEncoder
from molgraph.chemistry import MolecularGraphEncoder3D
from molgraph.chemistry import ConformerGenerator
from molgraph.chemistry import Featurizer
from molgraph.chemistry import Tokenizer
from molgraph.chemistry import features


# Define atomic encoders
atom_encoder = Featurizer([
    features.Symbol({'C', 'N', 'O', 'P', 'Na'}),
    features.Hybridization(),
])
bond_encoder = Featurizer([
    features.BondType(),
    features.Rotatable()
])
# Define molecular graph encoder
encoder = MolecularGraphEncoder(atom_encoder, bond_encoder)

# Typical graph, with nested ragged tensors
input_1a = encoder(['OCC1OC(C(C1O)O)n1cnc2c1ncnc2N', 'C(C(=O)O)N'])
# Typical graph, with nested tensors
input_2a = input_1a.merge()
# Typical graph, with nested ragged tensors, without edge features
input_3a = input_1a.remove(['edge_feature'])
# Typical graph, with nested tensors, without edge features
input_4a = input_3a.merge()
# Single node graph, with nested tensors
input_5a = encoder(['C']).merge()
# Single node graph, with nested tensors, without edge features
input_6a = input_5a.remove(['edge_feature'])
# Disconnected graph (first node is disconnected), with nested tensors
input_7a = encoder(['[Na+].[O-]c1ccccc1']).merge()
# Disconnected graph (last node is disconnected), with nested tensors
input_8a = encoder(['[O-]c1ccccc1.[Na+]']).merge()

inputs = [
    input_1a, input_2a, input_3a, input_4a, 
    input_5a, input_6a, input_7a, input_8a,
]

# Define molecular graph encoder
encoder = MolecularGraphEncoder3D(
    atom_encoder, conformer_generator=ConformerGenerator())

# Typical graph, with nested ragged tensors
input_1b = encoder(['OCC1OC(C(C1O)O)n1cnc2c1ncnc2N', 'C(C(=O)O)N'])
# Typical graph, with nested tensors
input_2b = input_1b.merge()
# Single node graph, with nested tensors
input_5b = encoder(['C']).merge()
# Disconnected graph (first node is disconnected), with nested tensors
input_7b = encoder(['[Na+].[O-]c1ccccc1']).merge()
# Disconnected graph (last node is disconnected), with nested tensors
input_8b = encoder(['[O-]c1ccccc1.[Na+]']).merge()

inputs_3d = [
    input_1b, input_2b,           
    input_5b,           input_7b, input_8b,
]

# Define atomic encoders
atom_encoder = Tokenizer([
    features.Symbol({'C', 'N', 'O', 'P', 'Na'}),
    features.Hybridization(),
])
bond_encoder = Tokenizer([
    features.BondType(),
    features.Rotatable()
])
# Define molecular graph encoder
encoder = MolecularGraphEncoder(
    atom_encoder, bond_encoder)

# Typical graph, with nested ragged tensors
input_1c = encoder(['OCC1OC(C(C1O)O)n1cnc2c1ncnc2N', 'C(C(=O)O)N'])
# Typical graph, with nested tensors
input_2c = input_1c.merge()
# Typical graph, with nested ragged tensors, without edge features
input_3c = input_1c.remove(['edge_feature'])
# Typical graph, with nested tensors, without edge features
input_4c = input_3c.merge()
# Single node graph, with nested tensors
input_5c = encoder(['C']).merge()
# Single node graph, with nested tensors, without edge features
input_6c = input_5c.remove(['edge_feature'])
# Disconnected graph (first node is disconnected), with nested tensors
input_7c = encoder(['[Na+].[O-]c1ccccc1']).merge()
# Disconnected graph (last node is disconnected), with nested tensors
input_8c = encoder(['[O-]c1ccccc1.[Na+]']).merge()

inputs_tokenized = [
    input_1c, input_2c, input_3c, input_4c, 
    input_5c, input_6c, input_7c, input_8c,
]


class BaseLayerTestCase(unittest.TestCase):
    
    def _test_output_shape(self, layer, parameters):

        for param in parameters:

            for inp in inputs:

                layer_instance1 = layer(**param)
                layer_instance2 = layer(**param)
                
                x1 = layer_instance1(inp)

                units = layer_instance1.units 

                self.assertEqual(x1.node_feature.shape[0], inp.shape[0])
                self.assertEqual(x1.node_feature.shape[-1], units)

                x2 = layer_instance2(x1)

                self.assertEqual(x2.node_feature.shape[0], inp.shape[0])
                self.assertEqual(x2.node_feature.shape[-1], units)
                
                if layer_instance1.update_edge_features and x1.edge_feature is not None:
                    self.assertEqual(x2.edge_feature.shape[0], inp.edge_feature.shape[0])
                    self.assertEqual(x2.edge_feature.shape[-1], units)

    def _test_output_value(self, layer, parameters):

        for param in parameters:

            for inp in inputs:

                layer_instance1 = layer(**param)
                layer_instance2 = layer(**param)
                
                x1 = layer_instance1(inp)
                x2 = layer_instance2(x1)

                if x1.node_feature.shape[0] > 1:
                    result = tf.math.equal(x1.node_feature, x2.node_feature)
                    self.assertFalse(tf.math.reduce_all(result))

                if (
                    layer_instance1.update_edge_features and 
                    x1.edge_feature is not None and 
                    x1.edge_feature.shape[0] != 0
                ):
                    result = tf.math.equal(x1.edge_feature, x2.edge_feature)
                    self.assertFalse(tf.math.reduce_all(result))



class BaseGeometricLayerTestCase(unittest.TestCase):
    
    def _test_output_shape(self, layer, parameters):

        for param in parameters:

            for inp in inputs_3d:

                layer_instance1 = layer(**param)
                layer_instance2 = layer(**param)
                
                x1 = layer_instance1(inp)

                units = layer_instance1.units 

                self.assertEqual(x1.node_feature.shape[0], inp.shape[0])
                self.assertEqual(x1.node_feature.shape[-1], units)

                x2 = layer_instance2(x1)

                self.assertEqual(x2.node_feature.shape[0], inp.shape[0])
                self.assertEqual(x2.node_feature.shape[-1], units)
                

    def _test_output_value(self, layer, parameters):

        for param in parameters:

            for inp in inputs_3d:

                layer_instance1 = layer(**param)
                layer_instance2 = layer(**param)
                
                x1 = layer_instance1(inp)
                x2 = layer_instance2(x1)

                if x1.node_feature.shape[0] > 1:
                    result = tf.math.equal(x1.node_feature, x2.node_feature)
                    self.assertFalse(tf.math.reduce_all(result))


