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
    'OCC1OC(C(C1O)O)n1cnc2c1ncnc2N', 'C(C(=O)O)N', '[Na+].[O-]c1ccccc1'])


@pytest.mark.parametrize('graph_tensor', [graph_tensor])
def test_shape_and_dtype(graph_tensor) -> None:
    graph_tensor.shape.assert_is_compatible_with(
        tf.TensorShape([3, None, 11]))
    graph_tensor.node_feature.shape.assert_is_compatible_with(
        tf.TensorShape([3, None, 11]))
    graph_tensor.edge_feature.shape.assert_is_compatible_with(
        tf.TensorShape([3, None, 5]))
    graph_tensor.edge_dst.shape.assert_is_compatible_with(
        tf.TensorShape([3, None]))
    graph_tensor.edge_src.shape.assert_is_compatible_with(
        tf.TensorShape([3, None]))

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
        tf.TensorShape([32, 11]))
    graph_tensor_merged.node_feature.shape.assert_is_compatible_with(
        tf.TensorShape([32, 11]))
    graph_tensor_merged.edge_feature.shape.assert_is_compatible_with(
        tf.TensorShape([64, 5]))
    graph_tensor_merged.edge_dst.shape.assert_is_compatible_with(
        tf.TensorShape([64,]))
    graph_tensor_merged.edge_src.shape.assert_is_compatible_with(
        tf.TensorShape([64,]))

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
def test_incompatible_size_2(graph_tensor) -> None:
    with pytest.raises(tf.errors.InvalidArgumentError):
        graph_tensor_merged = graph_tensor.merge()
        graph_tensor_merged = graph_tensor_merged.update({
            'node_feature': tf.ragged.constant([[[1., 2.], [3., 4.]], [[2., 1.]]])
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
            'random_feature': tf.random.uniform((10, 5))
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
        x = x.update({'node_feature': tf.random.uniform((32, 11))})
        x = x.update({'random_feature': tf.random.uniform((32, 11))})
        x = x.remove(['random_feature'])
        return x

    graph_tensor_merged = f1(graph_tensor)
    graph_tensor_separated = f2(graph_tensor_merged)
    graph_tensor = f3(graph_tensor_separated)
    _ = f4(graph_tensor)

@pytest.mark.parametrize('graph_tensor', [graph_tensor])
def test_indexing(graph_tensor) -> None:
    graph_tensor_sliced = graph_tensor[[1, 2]]
    graph_tensor_sliced = graph_tensor_sliced.merge()
    graph_tensor_sliced.node_feature.shape.assert_is_compatible_with(
        tf.TensorShape([13, 11]))
    graph_tensor_sliced.edge_feature.shape.assert_is_compatible_with(
        tf.TensorShape([22, 5]))
    graph_tensor_sliced.edge_dst.shape.assert_is_compatible_with(
        tf.TensorShape([22,]))
    graph_tensor_sliced.edge_src.shape.assert_is_compatible_with(
        tf.TensorShape([22,]))

    graph_tensor_sliced = graph_tensor[1]
    graph_tensor_sliced.node_feature.shape.assert_is_compatible_with(
        tf.TensorShape([5, 11]))
    graph_tensor_sliced.edge_feature.shape.assert_is_compatible_with(
        tf.TensorShape([8, 5]))
    graph_tensor_sliced.edge_dst.shape.assert_is_compatible_with(
        tf.TensorShape([8,]))
    graph_tensor_sliced.edge_src.shape.assert_is_compatible_with(
        tf.TensorShape([8,]))

    graph_tensor_sliced = graph_tensor.merge()[[1, 2]]
    graph_tensor_sliced.node_feature.shape.assert_is_compatible_with(
        tf.TensorShape([13, 11]))
    graph_tensor_sliced.edge_feature.shape.assert_is_compatible_with(
        tf.TensorShape([22, 5]))
    graph_tensor_sliced.edge_dst.shape.assert_is_compatible_with(
        tf.TensorShape([22,]))
    graph_tensor_sliced.edge_src.shape.assert_is_compatible_with(
        tf.TensorShape([22,]))

    graph_tensor_sliced = graph_tensor.merge()[1]
    graph_tensor_sliced.node_feature.shape.assert_is_compatible_with(
        tf.TensorShape([5, 11]))
    graph_tensor_sliced.edge_feature.shape.assert_is_compatible_with(
        tf.TensorShape([8, 5]))
    graph_tensor_sliced.edge_dst.shape.assert_is_compatible_with(
        tf.TensorShape([8,]))
    graph_tensor_sliced.edge_src.shape.assert_is_compatible_with(
        tf.TensorShape([8,]))

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
    assert not tf.reduce_all(graph_tensor.node_feature == node_feature_merged)

@pytest.mark.parametrize('graph_tensor', [graph_tensor])
def test_update_method(graph_tensor) -> None:

    GraphTensor(
        data={
            'edge_dst': (0, 0, 1, 2, 3, 2),
            'edge_src': (1, 2, 0, 0, 2, 3),
            'node_feature': ((0., 1., 0.), (1., 0., 1.), (0., 2., 1.), (3., 0., 0.)),
        }
    )

    GraphTensor(
        data={
            'edge_dst': [0, 0, 1, 2, 3, 2],
            'edge_src': [1, 2, 0, 0, 2, 3],
            'node_feature': [[0., 1., 0.], [1., 0., 1.], [0., 2., 1.], [3., 0., 0.]],
        }
    )

    GraphTensor(
        data={
            'edge_dst': np.array([0, 0, 1, 2, 3, 2]),
            'edge_src': np.array([1, 2, 0, 0, 2, 3]),
            'node_feature': np.array([[0., 1., 0.], [1., 0., 1.], [0., 2., 1.], [3., 0., 0.]]),
        }
    )

    GraphTensor(
        data={
            'edge_dst': tf.constant([0, 0, 1, 2, 3, 2]),
            'edge_src': tf.constant([1, 2, 0, 0, 2, 3]),
            'node_feature': tf.constant([[0., 1., 0.], [1., 0., 1.], [0., 2., 1.], [3., 0., 0.]]),
        }
    )

    GraphTensor(
        data={
            'edge_dst': [
                (0, 0, 1, 2, 3, 2),
                [0, 1, 2, 1]
            ],
            'edge_src': (
                (1, 2, 0, 0, 2, 3),
                (1, 0, 1, 2)
            ),
            'node_feature': [
                [(0., 1., 0.), [1., 0., 1.], [0., 2., 1.], [3., 0., 0.]],
                [(1., 1., 1.), [0., 5., 1.], [5., 4., 3.]]
            ]
        }
    )

    GraphTensor(
        data={
            'edge_dst': [
                [0, 0, 1, 2, 3, 2],
                [0, 1, 2, 1]
            ],
            'edge_src': [
                [1, 2, 0, 0, 2, 3],
                [1, 0, 1, 2]
            ],
            'node_feature': [
                [[0., 1., 0.], [1., 0., 1.], [0., 2., 1.], [3., 0., 0.]],
                [[1., 1., 1.], [0., 5., 1.], [5., 4., 3.]]
            ]
        }
    )

    GraphTensor(
        data={
            'edge_dst': np.array([
                [0, 0, 1, 2, 3, 2],
                [0, 1, 2, 1]
            ], dtype=object),
            'edge_src': np.array([
                [1, 2, 0, 0, 2, 3],
                [1, 0, 1, 2]
            ], dtype=object),
            'node_feature': np.array([
                [[0., 1., 0.], [1., 0., 1.], [0., 2., 1.], [3., 0., 0.]],
                [[1., 1., 1.], [0., 5., 1.], [5., 4., 3.]]
            ], dtype=object)
        }
    )

    GraphTensor(
        data={
            'edge_dst': tf.ragged.constant([
                [0, 0, 1, 2, 3, 2],
                [0, 1, 2, 1]
            ], ragged_rank=1),
            'edge_src': tf.ragged.constant([
                [1, 2, 0, 0, 2, 3],
                [1, 0, 1, 2]
            ], ragged_rank=1),
            'node_feature': tf.ragged.constant([
                [[0., 1., 0.], [1., 0., 1.], [0., 2., 1.], [3., 0., 0.]],
                [[1., 1., 1.], [0., 5., 1.], [5., 4., 3.]]
            ], ragged_rank=1)
        }
    )

# @pytest.mark.parametrize('graph_tensor', [graph_tensor])
# def test_remove_method(graph_tensor) -> None:
#     graph_tensor = graph_tensor.remove(['edge_feature', 'positional_encoding'])
#     assert not hasattr(graph_tensor, 'edge_feature')
#     assert not hasattr(graph_tensor, 'positional_encoding')

@pytest.mark.parametrize('graph_tensor', [graph_tensor])
def test_dataset(graph_tensor) -> None:
    ds = tf.data.Dataset.from_tensor_slices(graph_tensor)
    ds_1 = ds.batch(2).take(1)
    for x in ds_1: pass

    x.shape.assert_is_compatible_with(
        tf.TensorShape([2, None, 11]))
    x.node_feature.shape.assert_is_compatible_with(
        tf.TensorShape([2, None, 11]))
    x.edge_feature.shape.assert_is_compatible_with(
        tf.TensorShape([2, None, 5]))
    x.edge_dst.shape.assert_is_compatible_with(
        tf.TensorShape([2, None]))
    x.edge_src.shape.assert_is_compatible_with(
        tf.TensorShape([2, None]))

    ds_2 = ds.batch(2).take(1).map(lambda x: x.merge())
    for x in ds_2: pass

    x.shape.assert_is_compatible_with(
        tf.TensorShape([24, 11]))
    x.node_feature.shape.assert_is_compatible_with(
        tf.TensorShape([24, 11]))
    x.edge_feature.shape.assert_is_compatible_with(
        tf.TensorShape([50, 5]))
    x.edge_dst.shape.assert_is_compatible_with(
        tf.TensorShape([50,]))
    x.edge_src.shape.assert_is_compatible_with(
        tf.TensorShape([50,]))
