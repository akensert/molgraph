import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import numpy as np

from molgraph import layers
from molgraph.chemistry.molecular_encoders import MolecularGraphEncoder
from molgraph.chemistry.atomic.featurizers import AtomFeaturizer
from molgraph.chemistry.atomic.featurizers import BondFeaturizer
from molgraph.chemistry.atomic import features
from molgraph.tensors.graph_tensor import GraphTensor

import pytest

# Define atomic encoders
atom_encoder = AtomFeaturizer([
    features.Symbol({'C', 'N', 'O'}, oov_size=1),
    features.Hybridization({'SP', 'SP2', 'SP3'}, oov_size=1),
    features.HydrogenDonor(),
    features.HydrogenAcceptor(),
    features.Hetero()
])
bond_encoder = BondFeaturizer([
    features.BondType({'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'}),
    features.Rotatable()
])
# Define molecular graph encoder
encoder = MolecularGraphEncoder(atom_encoder, bond_encoder)

# Typical graph, with nested ragged tensors
graph_tensor = encoder([
    'C'])


@pytest.mark.parametrize('graph_tensor', [graph_tensor])
def test_shape_and_dtype(graph_tensor) -> None:
    graph_tensor.shape.assert_is_compatible_with(
        tf.TensorShape([1, None, 11]))
    graph_tensor.node_feature.shape.assert_is_compatible_with(
        tf.TensorShape([1, None, 11]))
    graph_tensor.edge_feature.shape.assert_is_compatible_with(
        tf.TensorShape([1, None, 5]))
    graph_tensor.edge_dst.shape.assert_is_compatible_with(
        tf.TensorShape([1, None]))
    graph_tensor.edge_src.shape.assert_is_compatible_with(
        tf.TensorShape([1, None]))

    assert graph_tensor.dtype.name == 'float32'
    assert graph_tensor.node_feature.dtype.name == 'float32'
    assert graph_tensor.edge_feature.dtype.name == 'float32'
    assert graph_tensor.edge_dst.dtype.name == 'int32'
    assert graph_tensor.edge_src.dtype.name == 'int32'

    graph_tensor = graph_tensor.merge()

    assert graph_tensor.graph_indicator.dtype.name == 'int32'

@pytest.mark.parametrize('graph_tensor', [graph_tensor])
def test_merge_and_separate(graph_tensor) -> None:
    graph_tensor_merged = graph_tensor.merge()
    graph_tensor_merged.shape.assert_has_rank(2)
    graph_tensor_merged.shape.assert_is_compatible_with(
        tf.TensorShape([1, 11]))
    graph_tensor_merged.node_feature.shape.assert_is_compatible_with(
        tf.TensorShape([1, 11]))
    graph_tensor_merged.edge_feature.shape.assert_is_compatible_with(
        tf.TensorShape([0, 5]))
    graph_tensor_merged.edge_dst.shape.assert_is_compatible_with(
        tf.TensorShape([0,]))
    graph_tensor_merged.edge_src.shape.assert_is_compatible_with(
        tf.TensorShape([0,]))

    assert graph_tensor_merged.dtype.name == 'float32'
    assert graph_tensor_merged.node_feature.dtype.name == 'float32'
    assert graph_tensor_merged.edge_feature.dtype.name == 'float32'
    assert graph_tensor_merged.edge_dst.dtype.name == 'int32'
    assert graph_tensor_merged.edge_src.dtype.name == 'int32'
    assert graph_tensor_merged.graph_indicator.dtype.name == 'int32'

    graph_tensor_separated = graph_tensor_merged.separate()

    graph_tensor_separated.shape.assert_has_rank(3)

    graph_tensor.node_feature.shape.assert_is_compatible_with(
        graph_tensor_separated.node_feature.shape
    )
    graph_tensor.edge_feature.shape.assert_is_compatible_with(
        graph_tensor_separated.edge_feature.shape
    )
    graph_tensor.edge_dst.shape.assert_is_compatible_with(
        graph_tensor_separated.edge_dst.shape
    )
    graph_tensor.edge_src.shape.assert_is_compatible_with(
        graph_tensor_separated.edge_src.shape
    )


@pytest.mark.parametrize('graph_tensor', [graph_tensor])
def test_incompatible_size(graph_tensor) -> None:
    with pytest.raises(tf.errors.InvalidArgumentError):
        graph_tensor_merged = graph_tensor.merge()
        graph_tensor_merged = graph_tensor_merged.update({
            'random_feature': tf.random.uniform((3, 3))
        })

@pytest.mark.parametrize('graph_tensor', [graph_tensor])
def test_different_tensor_type_but_compatible_size(graph_tensor) -> None:
    graph_tensor_merged = graph_tensor.merge()
    random_feature = tf.random.uniform(graph_tensor_merged.node_feature.shape)
    value_rowids = graph_tensor_merged.graph_indicator
    graph_tensor_merged = graph_tensor_merged.update({
        'random_feature': tf.RaggedTensor.from_value_rowids(
            random_feature, value_rowids
        )
    })

@pytest.mark.parametrize('graph_tensor', [graph_tensor])
def test_incompatible_tensor_type_for_replacement(graph_tensor) -> None:
    with pytest.raises(tf.errors.InvalidArgumentError):
        graph_tensor_merged = graph_tensor.merge()
        random_feature = tf.random.uniform(graph_tensor_merged.node_feature.shape)
        value_rowids = graph_tensor_merged.graph_indicator
        graph_tensor_merged = graph_tensor_merged.update({
            'random_feature': tf.RaggedTensor.from_value_rowids(
                random_feature, value_rowids
            )
        })
        graph_tensor_merged = graph_tensor_merged.update({
            'random_feature': tf.random.uniform((2, 5))
        })

@pytest.mark.parametrize('graph_tensor', [graph_tensor])
def test_incompatible_tensor_type_for_replacement_reversed(graph_tensor) -> None:

    graph_tensor_merged = graph_tensor.merge()
    random_feature = tf.random.uniform(graph_tensor_merged.node_feature.shape)
    graph_tensor_merged = graph_tensor_merged.update({
        'random_feature': random_feature})

    with pytest.raises(tf.errors.InvalidArgumentError):
        random_feature = tf.ragged.constant([[1., 2.], [4., 5., 6., 7.]])
        graph_tensor_merged = graph_tensor_merged.update({
            'random_feature': random_feature
        })

@pytest.mark.parametrize('graph_tensor', [graph_tensor])
def test_compatible_tf_function(graph_tensor) -> None:

    @tf.function
    def f1(x):
        return x.merge()

    @tf.function
    def f2(x):
        return x.separate()

    @tf.function
    def f3(x):
        return x.merge().separate().merge()

    @tf.function
    def f4(x):
        x = x.update({'node_feature': tf.random.uniform((1, 11))})
        x = x.update({'random_feature': tf.random.uniform((1, 11))})
        x = x.remove(['random_feature'])
        return x

    graph_tensor_merged = f1(graph_tensor)
    graph_tensor_separated = f2(graph_tensor_merged)
    graph_tensor = f3(graph_tensor_separated)
    _ = f4(graph_tensor)

@pytest.mark.parametrize('graph_tensor', [graph_tensor])
def test_indexing(graph_tensor) -> None:
    graph_tensor_sliced = graph_tensor[[0]]
    graph_tensor_sliced = graph_tensor_sliced.merge()
    graph_tensor_sliced.node_feature.shape.assert_is_compatible_with(
        tf.TensorShape([1, 11]))
    graph_tensor_sliced.edge_feature.shape.assert_is_compatible_with(
        tf.TensorShape([0, 5]))
    graph_tensor_sliced.edge_dst.shape.assert_is_compatible_with(
        tf.TensorShape([0,]))
    graph_tensor_sliced.edge_src.shape.assert_is_compatible_with(
        tf.TensorShape([0,]))

    graph_tensor_sliced = graph_tensor[0]
    graph_tensor_sliced.node_feature.shape.assert_is_compatible_with(
        tf.TensorShape([1, 11]))
    graph_tensor_sliced.edge_feature.shape.assert_is_compatible_with(
        tf.TensorShape([0, 5]))
    graph_tensor_sliced.edge_dst.shape.assert_is_compatible_with(
        tf.TensorShape([0,]))
    graph_tensor_sliced.edge_src.shape.assert_is_compatible_with(
        tf.TensorShape([0,]))

    graph_tensor_sliced = graph_tensor.merge()[[0]]
    graph_tensor_sliced.node_feature.shape.assert_is_compatible_with(
        tf.TensorShape([1, 11]))
    graph_tensor_sliced.edge_feature.shape.assert_is_compatible_with(
        tf.TensorShape([0, 5]))
    graph_tensor_sliced.edge_dst.shape.assert_is_compatible_with(
        tf.TensorShape([0,]))
    graph_tensor_sliced.edge_src.shape.assert_is_compatible_with(
        tf.TensorShape([0,]))

@pytest.mark.parametrize('graph_tensor', [graph_tensor])
def test_spec(graph_tensor) -> None:
    spec = graph_tensor.spec
    for key in spec._data_spec.keys():
        assert isinstance(spec._data_spec[key], tf.RaggedTensorSpec)
        assert spec._data_spec[key].ragged_rank == 1
        spec._data_spec[key].shape.assert_is_compatible_with(
            graph_tensor[key].shape)
        assert spec._data_spec[key].dtype.name == graph_tensor[key].dtype.name

    graph_tensor = graph_tensor.merge()
    spec = graph_tensor.spec
    for key in spec._data_spec.keys():
        assert isinstance(spec._data_spec[key], tf.TensorSpec)
        spec._data_spec[key].shape.assert_is_compatible_with(
            graph_tensor[key].shape)
        assert spec._data_spec[key].dtype.name == graph_tensor[key].dtype.name

@pytest.mark.parametrize('graph_tensor', [graph_tensor])
def test_unspecific_spec(graph_tensor) -> None:
    spec = graph_tensor.unspecific_spec
    for key in spec._data_spec.keys():
        assert isinstance(spec._data_spec[key], tf.RaggedTensorSpec)
        assert spec._data_spec[key].ragged_rank == 1
        spec._data_spec[key].shape.assert_is_compatible_with(
            graph_tensor[key].shape)
        assert spec._data_spec[key].dtype.name == graph_tensor[key].dtype.name

    graph_tensor = graph_tensor.merge()
    spec = graph_tensor.unspecific_spec
    for key in spec._data_spec.keys():
        assert isinstance(spec._data_spec[key], tf.TensorSpec)
        spec._data_spec[key].shape.assert_is_compatible_with(
            graph_tensor[key].shape)
        assert spec._data_spec[key].dtype.name == graph_tensor[key].dtype.name


@pytest.mark.parametrize('graph_tensor', [graph_tensor])
def test_update_method(graph_tensor) -> None:

    node_feature = graph_tensor.node_feature

    graph_tensor_merged = graph_tensor.merge()

    node_feature_merged = graph_tensor_merged.node_feature

    graph_tensor = graph_tensor.update({
        'node_feature': node_feature_merged})

    assert not (
        graph_tensor.node_feature.shape.as_list() ==
        node_feature_merged.shape.as_list()
    )
    assert (
        graph_tensor.node_feature.shape.as_list() ==
        node_feature.shape.as_list()
    )

    graph_tensor_merged = graph_tensor_merged.update({
        'node_feature': tf.random.uniform(
            shape=node_feature_merged.shape)})

    assert not type(graph_tensor.node_feature) == type(node_feature_merged)

# @pytest.mark.parametrize('graph_tensor', [graph_tensor])
# def test_remove_method(graph_tensor) -> None:
#     graph_tensor = graph_tensor.remove(['edge_feature', 'positional_encoding'])
#     assert not hasattr(graph_tensor, 'edge_feature')
#     assert not hasattr(graph_tensor, 'positional_encoding')

@pytest.mark.parametrize('graph_tensor', [graph_tensor])
def test_dataset(graph_tensor) -> None:
    ds = tf.data.Dataset.from_tensor_slices(graph_tensor)
    ds_1 = ds.batch(1).take(1)
    for x in ds_1: pass

    x.shape.assert_is_compatible_with(
        tf.TensorShape([1, None, 11]))
    x.node_feature.shape.assert_is_compatible_with(
        tf.TensorShape([1, None, 11]))
    x.edge_feature.shape.assert_is_compatible_with(
        tf.TensorShape([1, None, 5]))
    x.edge_dst.shape.assert_is_compatible_with(
        tf.TensorShape([1, None]))
    x.edge_src.shape.assert_is_compatible_with(
        tf.TensorShape([1, None]))

    ds_2 = ds.batch(1).take(1).map(lambda x: x.merge())
    for x in ds_2: pass

    x.shape.assert_is_compatible_with(
        tf.TensorShape([1, 11]))
    x.node_feature.shape.assert_is_compatible_with(
        tf.TensorShape([1, 11]))
    x.edge_feature.shape.assert_is_compatible_with(
        tf.TensorShape([0, 5]))
    x.edge_dst.shape.assert_is_compatible_with(
        tf.TensorShape([0,]))
    x.edge_src.shape.assert_is_compatible_with(
        tf.TensorShape([0,]))
