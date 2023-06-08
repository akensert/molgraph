import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from functools import partial

import sys
sys.path.append('../')

from molgraph import layers
from molgraph.chemistry import MolecularGraphEncoder
from molgraph.chemistry import Featurizer
from molgraph.chemistry import features

import pytest



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
input_1 = encoder(['OCC1OC(C(C1O)O)n1cnc2c1ncnc2N', 'C(C(=O)O)N'])
# Typical graph, with nested tensors
input_2 = input_1.merge()
# Typical graph, with nested ragged tensors, without edge features
input_3 = input_1.remove(['edge_feature'])
# Typical graph, with nested tensors, without edge features
input_4 = input_3.merge()
# Single node graph, with nested tensors
input_5 = encoder(['C']).merge()
# Single node graph, with nested tensors, without edge features
input_6 = input_5.remove(['edge_feature'])
# Disconnected graph (first node is disconnected), with nested tensors
input_7 = encoder(['[Na+].[O-]c1ccccc1']).merge()
# Disconnected graph (last node is disconnected), with nested tensors
input_8 = encoder(['[O-]c1ccccc1.[Na+]']).merge()

inputs = [
    input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8,
]


def map_fn(inp, layer, parameters):
    # two-layered GNN
    layer1 = layer(**parameters)
    layer2 = layer(**parameters)
    out = layer1(inp)
    out = layer2(out)
    units = layer2.units
    assert out.shape[0] == inp.shape[0]
    assert out.shape[-1] == units
    if layer1.update_edge_features and hasattr(inp, 'edge_feature'):
        assert out.edge_feature.shape[0] == inp.edge_feature.shape[0]
        assert out.edge_feature.shape[-1] == units


@pytest.mark.parametrize("parameters", [
    {'units': None,
    'residual': False,
    'merge_mode': 'mean', 
    'self_projection': False},
    {'units': 128, 'residual': False, 'self_projection': False},
    {'units': None, 'residual': True, 'merge_mode': 'mean', 'self_projection': True},
    {'units': 128, 'residual': True, 'self_projection': True},
    {'units': 128, 'apply_initial_node_projection': True, 'residual': True, 'self_projection': True},
    {'units': 128, 'apply_initial_node_projection': False, 'residual': True, 'self_projection': True},
])
def test_attentive_fp_conv(parameters) -> None:
    list(map(partial(map_fn, layer=layers.AttentiveFPConv, parameters=parameters), inputs))

@pytest.mark.parametrize("parameters", [
    {'units': None,
    'residual': False,
    'self_projection': False},
    {'units': 128, 'residual': False, 'self_projection': False},
    {'units': None, 'residual': True, 'self_projection': True},
    {'units': 128, 'residual': True, 'self_projection': True},
    {'units': 128, 'use_edge_features': True, 'residual': True, 'self_projection': True},
    {'units': 128, 'use_edge_features': False, 'residual': True, 'self_projection': True},
])
def test_gcnii_conv(parameters) -> None:
    list(map(partial(map_fn, layer=layers.GCNIIConv, parameters=parameters), inputs))

@pytest.mark.parametrize("parameters", [
    {'units': None,
    'residual': False,
    'self_projection': False},
    {'units': 128, 'residual': False, 'self_projection': False},
    {'units': None, 'residual': True, 'self_projection': True},
    {'units': 128, 'residual': True, 'self_projection': True},
    {'units': 128, 'use_edge_features': True, 'residual': True, 'self_projection': True},
    {'units': None, 'use_edge_features': True, 'residual': True, 'self_projection': True},
    {'units': 128, 'use_edge_features': True, 'residual': True, 'self_projection': True, 'num_bases': 3},
    {'units': 128, 'use_edge_features': True, 'residual': True, 'self_projection': True, 'num_bases': 999},
])
def test_gcn_conv(parameters) -> None:
    list(map(partial(map_fn, layer=layers.GCNConv, parameters=parameters), inputs))

@pytest.mark.parametrize("parameters", [
    {'units': None, 'residual': False, 'self_projection': False},
    {'units': 128, 'residual': False, 'self_projection': False},
    {'units': None, 'residual': True, 'self_projection': True},
    {'units': 128, 'residual': True, 'self_projection': True},
    {'units': 128, 'use_edge_features': True, 'residual': True, 'self_projection': True},
])
def test_gin_conv(parameters) -> None:
    list(map(partial(map_fn, layer=layers.GINConv, parameters=parameters), inputs))

@pytest.mark.parametrize("parameters", [
    {'units': None, 'residual': False, 'self_projection': False},
    {'units': 128, 'residual': False, 'self_projection': False},
    {'units': None, 'residual': True, 'self_projection': True},
    {'units': 128, 'residual': True, 'self_projection': True},
    {'units': 128, 'residual': True, 'self_projection': True, 'aggregation_mode': 'mean'},
    {'units': 128, 'residual': True, 'self_projection': True, 'aggregation_mode': 'max'},
    {'units': 128, 'residual': True, 'self_projection': True, 'aggregation_mode': 'lstm'},
])
def test_graph_sage_conv(parameters) -> None:
    list(map(partial(map_fn, layer=layers.GraphSageConv, parameters=parameters), inputs))

@pytest.mark.parametrize("parameters", [
    {'units': None, 'residual': False, 'self_projection': False, 'merge_mode': 'mean'},
    {'units': 128, 'residual': False, 'self_projection': False},
    {'units': None, 'residual': True, 'self_projection': True, 'merge_mode': 'mean'},
    {'units': 128, 'residual': True, 'self_projection': True},
    {'units': 128, 'use_edge_features': True, 'residual': True, 'self_projection': True},
    {'units': None, 'use_edge_features': True, 'residual': True, 'self_projection': True, 'merge_mode': 'mean'},
    {'units': 128, 'use_edge_features': True, 'residual': True, 'self_projection': True, 'num_heads': 1, 'merge_mode': 'concat'},
    {'units': 128, 'use_edge_features': True, 'residual': True, 'self_projection': True, 'num_heads': 16, 'merge_mode': 'concat'},
    {'units': 128, 'use_edge_features': True, 'residual': True, 'self_projection': True, 'num_heads': 8, 'merge_mode': 'mean'},
    {'units': None, 'use_edge_features': True, 'residual': True, 'self_projection': True, 'num_heads': 8, 'merge_mode': 'mean'},
    {'units': 33, 'use_edge_features': True, 'residual': True, 'self_projection': True, 'num_heads': 8, 'merge_mode': 'mean'},
])
def test_gat_conv(parameters) -> None:
    list(map(partial(map_fn, layer=layers.GATConv, parameters=parameters), inputs))

@pytest.mark.parametrize("parameters", [
    {'units': None, 'residual': False, 'self_projection': False, 'merge_mode': 'mean'},
    {'units': 128, 'residual': False, 'self_projection': False},
    {'units': None, 'residual': True, 'self_projection': True, 'merge_mode': 'mean'},
    {'units': 128, 'residual': True, 'self_projection': True},
    {'units': 128, 'use_edge_features': True, 'residual': True, 'self_projection': True},
    {'units': None, 'use_edge_features': True, 'residual': True, 'self_projection': True, 'merge_mode': 'mean'},
    {'units': 128, 'use_edge_features': True, 'residual': True, 'self_projection': True, 'num_heads': 1, 'merge_mode': 'concat'},
    {'units': 128, 'use_edge_features': True, 'residual': True, 'self_projection': True, 'num_heads': 16, 'merge_mode': 'concat'},
    {'units': 128, 'use_edge_features': True, 'residual': True, 'self_projection': True, 'num_heads': 8, 'merge_mode': 'mean'},
    {'units': None, 'use_edge_features': True, 'residual': True, 'self_projection': True, 'num_heads': 8, 'merge_mode': 'mean'},
    {'units': 33, 'use_edge_features': True, 'residual': True, 'self_projection': True, 'num_heads': 8, 'merge_mode': 'mean'},
])
def test_gatv2_conv(parameters) -> None:
    list(map(partial(map_fn, layer=layers.GATv2Conv, parameters=parameters), inputs))

@pytest.mark.parametrize("parameters", [
    {'units': None, 'residual': False, 'self_projection': False, 'merge_mode': 'mean'},
    {'units': 128, 'residual': False, 'self_projection': False},
    {'units': None, 'residual': True, 'self_projection': True, 'merge_mode': 'mean'},
    {'units': 128, 'residual': True, 'self_projection': True},
    {'units': 128, 'use_edge_features': True, 'residual': True, 'self_projection': True},
    {'units': None, 'use_edge_features': True, 'residual': True, 'self_projection': True, 'merge_mode': 'mean'},
    {'units': 128, 'use_edge_features': True, 'residual': True, 'self_projection': True, 'num_heads': 1, 'merge_mode': 'concat'},
    {'units': 128, 'use_edge_features': True, 'residual': True, 'self_projection': True, 'num_heads': 16, 'merge_mode': 'concat'},
    {'units': 128, 'use_edge_features': True, 'residual': True, 'self_projection': True, 'num_heads': 8, 'merge_mode': 'mean'},
    {'units': None, 'use_edge_features': True, 'residual': True, 'self_projection': True, 'num_heads': 8, 'merge_mode': 'mean'},
    {'units': 33, 'use_edge_features': True, 'residual': True, 'self_projection': True, 'num_heads': 8, 'merge_mode': 'mean'},
])
def test_graph_transformer_conv(parameters) -> None:
    list(map(partial(map_fn, layer=layers.GraphTransformerConv, parameters=parameters), inputs))

@pytest.mark.parametrize("parameters", [
    {'units': None, 'residual': False, 'self_projection': False, 'merge_mode': 'sum'},
    {'units': 128, 'residual': False, 'self_projection': False},
    {'units': None, 'residual': True, 'self_projection': True, 'merge_mode': 'sum'},
    {'units': 128, 'residual': True, 'self_projection': True},
    {'units': 128, 'use_edge_features': True, 'residual': True, 'self_projection': True},
    {'units': None, 'use_edge_features': True, 'residual': True, 'self_projection': True, 'merge_mode': 'mean'},
    {'units': 128, 'use_edge_features': True, 'residual': True, 'self_projection': True, 'num_kernels': 1, 'merge_mode': 'concat'},
    {'units': 128, 'use_edge_features': True, 'residual': True, 'self_projection': True, 'num_kernels': 16, 'merge_mode': 'concat'},
    {'units': 128, 'use_edge_features': True, 'residual': True, 'self_projection': True, 'num_kernels': 8, 'merge_mode': 'sum'},
    {'units': None, 'use_edge_features': True, 'residual': True, 'self_projection': True, 'num_kernels': 8, 'merge_mode': 'sum'},
    {'units': 33, 'use_edge_features': True, 'residual': True, 'self_projection': True, 'num_kernels': 8, 'merge_mode': 'sum', 'pseudo_coord_dim': 2},
    {'units': 33, 'use_edge_features': True, 'residual': True, 'self_projection': True, 'num_kernels': 8, 'merge_mode': 'sum', 'pseudo_coord_dim': 1},
    {'units': 33, 'use_edge_features': True, 'residual': True, 'self_projection': True, 'num_kernels': 8, 'merge_mode': 'sum', 'pseudo_coord_dim': 16},
])
def test_gmm_conv(parameters) -> None:
    list(map(partial(map_fn, layer=layers.GMMConv, parameters=parameters), inputs))

@pytest.mark.parametrize("parameters", [
    {'units': None,
    'residual': False,
    'self_projection': False},
    {'units': 128, 'residual': False, 'self_projection': False},
    {'units': None, 'residual': True, 'self_projection': True},
    {'units': 128, 'residual': True, 'self_projection': True},
    {'units': 128, 'use_edge_features': True, 'residual': True, 'self_projection': True},
    {'units': 128, 'use_edge_features': False, 'residual': True, 'self_projection': True},
])
def test_gated_gcn_conv(parameters) -> None:
    list(map(partial(map_fn, layer=layers.GatedGCNConv, parameters=parameters), inputs))

@pytest.mark.parametrize("parameters", [
    {'units': None,
    'residual': False,
    'self_projection': False},
    {'units': 128, 'residual': False, 'self_projection': False},
    {'units': None, 'residual': True, 'self_projection': True},
    {'units': 128, 'residual': True, 'self_projection': True},
    {'units': None, 'residual': True, 'self_projection': False},
    {'units': None, 'residual': False, 'self_projection': False},
    {'units': None, 'update_mode': 'dense', 'residual': False, 'self_projection': False},
    {'units': 128, 'update_mode': 'dense', 'residual': False, 'self_projection': False},
])
def test_mpnn_conv(parameters) -> None:
    list(map(partial(map_fn, layer=layers.MPNNConv, parameters=parameters), inputs))


def map_fn_edge_conv(inp, layer, parameters):
    layer = layer(**parameters)
    out = layer(inp)
    units = layer.units
    assert out.shape[0] == inp.shape[0]
    assert out.edge_state.shape[-1] == units

@pytest.mark.parametrize("parameters", [
    {'units': 128, 'update_mode': 'GRU'},
    {'units': 128, 'update_mode': 'DENSE'},
    {'units': 33}
])
def test_edge_conv(parameters) -> None:
    list(map(partial(map_fn_edge_conv, layer=layers.EdgeConv, parameters=parameters), inputs[:4]))



# Define molecular graph encoder (without positional encoding)
encoder = MolecularGraphEncoder(atom_encoder, bond_encoder, positional_encoding_dim=None)

# Typical graph, with nested ragged tensors
input_1 = encoder(['OCC1OC(C(C1O)O)n1cnc2c1ncnc2N', 'C(C(=O)O)N'])
# Typical graph, with nested tensors
input_2 = input_1.merge()
# Typical graph, with nested ragged tensors, without edge features
input_3 = input_1.remove(['edge_feature'])
# Typical graph, with nested tensors, without edge features
input_4 = input_3.merge()
# Single node graph, with nested tensors
input_5 = encoder(['C']).merge()
# Single node graph, with nested tensors, without edge features
input_6 = input_5.remove(['edge_feature'])
# Disconnected graph (first node is disconnected), with nested tensors
input_7 = encoder(['[Na+].[O-]c1ccccc1']).merge()
# Disconnected graph (last node is disconnected), with nested tensors
input_8 = encoder(['[O-]c1ccccc1.[Na+]']).merge()

inputs_2 = [
    input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8,
]

def test_laplacian_positional_encoding_1() -> None:

    for inp in inputs:
        out = layers.LaplacianPositionalEncoding(dim=20)(inp)
        assert out.shape.as_list() == inp.shape.as_list()

def test_laplacian_positional_encoding_2() -> None:

    for inp in inputs_2:
        out = layers.LaplacianPositionalEncoding(dim=20)(inp)
        assert out.shape.as_list() == inp.shape.as_list()



# Test readout
# Define atomic encoders
atom_encoder = Featurizer([
    features.Symbol({'C', 'N', 'O', 'P', 'Na'}),
    features.Hybridization({'SP', 'SP2', 'SP3'}),
])
bond_encoder = Featurizer([
    features.BondType(),
    features.Rotatable()
])
# Define molecular graph encoder
encoder = MolecularGraphEncoder(atom_encoder, bond_encoder)

# Typical graph, with nested ragged tensors
input_1 = encoder([
    'OCC1OC(C(C1O)O)n1cnc2c1ncnc2N', 'C(C(=O)O)N', '[Na+].[O-]c1ccccc1'])
# Typical graph, with nested tensors
input_2 = input_1.merge()

inputs_3 = [
    input_1, input_2
]

def map_fn_2(inp, layer):
    out = layers.GCNConv(128)(inp)
    readout = layer()
    out = readout(out)

    assert out.shape[0] == 3
    if isinstance(readout, layers.SetGatherReadout):
        assert out.shape[-1] == 256
    else:
        assert out.shape[-1] == 128

    out = layers.GCNConv(None)(inp)
    out = layer()(out)

    assert out.shape[0] == 3
    if isinstance(readout, layers.SetGatherReadout):
        assert out.shape[-1] == 16
    else:
        assert out.shape[-1] == 8


def test_readout() -> None:
    list(map(partial(map_fn_2, layer=layers.Readout), inputs_3))

def test_transformer_readout() -> None:
    list(map(partial(map_fn_2, layer=layers.TransformerEncoderReadout), inputs_3))

def test_set_gather_readout() -> None:
    list(map(partial(map_fn_2, layer=layers.SetGatherReadout), inputs_3))

def test_attentive_fp_readout() -> None:
    list(map(partial(map_fn_2, layer=layers.AttentiveFPReadout), inputs_3))
