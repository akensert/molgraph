import numpy as np
import pandas as pd
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import tempfile
import shutil

from molgraph.chemistry import MolecularGraphEncoder
from molgraph.chemistry import Featurizer
from molgraph.chemistry import features
from molgraph.layers import GCNConv, Readout

from molgraph.models import GradientActivationMapping
from molgraph.models import SaliencyMapping
from molgraph.models import IntegratedSaliencyMapping
from molgraph.models import SmoothGradSaliencyMapping

import pytest

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

encoder = MolecularGraphEncoder(atom_encoder, bond_encoder)
encoder

graph_tensor = encoder([
    'OCC1OC(C(C1O)O)n1cnc2c1ncnc2N', 'C(C(=O)O)N', '[Na+].[O-]c1ccccc1',
    'C(C(=O)O)N', '[Na+].[O-]c1ccccc1'])


@pytest.mark.parametrize('graph_tensor', [graph_tensor])
def test_saliency_save_and_load(graph_tensor):

    # ragged
    sequential_model = tf.keras.Sequential([
        tf.keras.layers.Input(type_spec=graph_tensor.unspecific_spec),
        GCNConv(128, name='conv_1'),
        GCNConv(128, name='conv_2'),
        GCNConv(128, name='conv_3'),
        GCNConv(128, name='conv_4'),
        Readout(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    gam_model_1 = SaliencyMapping(
        sequential_model,
        'linear',
    )
    file = tempfile.NamedTemporaryFile()
    filename = file.name
    file.close()
    gam_model_1(graph_tensor)
    gam_model_1.save(filename)
    gam_model_1 = tf.keras.models.load_model(filename)
    gam_model_1.predict(graph_tensor, batch_size=2, verbose=1)
    shutil.rmtree(filename)

    gam_model_2 = IntegratedSaliencyMapping(
        sequential_model,
        'linear',
    )
    file = tempfile.NamedTemporaryFile()
    filename = file.name
    file.close()
    gam_model_2(graph_tensor)
    gam_model_2.save(filename)
    gam_model_2 = tf.keras.models.load_model(filename)
    gam_model_2.predict(graph_tensor, batch_size=2, verbose=1)
    shutil.rmtree(filename)

    gam_model_3 = SmoothGradSaliencyMapping(
        sequential_model,
        'linear',
    )
    file = tempfile.NamedTemporaryFile()
    filename = file.name
    file.close()
    gam_model_3(graph_tensor)
    gam_model_3.save(filename)
    gam_model_3 = tf.keras.models.load_model(filename)
    gam_model_3.predict(graph_tensor, batch_size=2, verbose=1)
    shutil.rmtree(filename)


@pytest.mark.parametrize('graph_tensor', [graph_tensor])
def test_gam_save_and_load(graph_tensor):

    # ragged
    sequential_model = tf.keras.Sequential([
        tf.keras.layers.Input(type_spec=graph_tensor.unspecific_spec),
        GCNConv(128, name='conv_1'),
        GCNConv(128, name='conv_2'),
        Readout(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    gam_model_1 = GradientActivationMapping(
        sequential_model,
        ['conv_1', 'conv_2'],
        'linear',
    )
    file = tempfile.NamedTemporaryFile()
    filename = file.name
    file.close()
    gam_model_1(graph_tensor)
    gam_model_1.save(filename)
    gam_model_1 = tf.keras.models.load_model(filename)
    gam_model_1.predict(graph_tensor, batch_size=2, verbose=1)
    shutil.rmtree(filename)


@pytest.mark.parametrize('graph_tensor', [graph_tensor])
def test_saliency_ragged(graph_tensor):

    # ragged
    sequential_model = tf.keras.Sequential([
        tf.keras.layers.Input(type_spec=graph_tensor.unspecific_spec),
        GCNConv(128, name='conv_1'),
        GCNConv(128, name='conv_2'),
        GCNConv(128, name='conv_3'),
        GCNConv(128, name='conv_4'),
        Readout(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    gam_model_1 = SaliencyMapping(
        sequential_model,
        'linear',
    )
    gam_model_1.predict(graph_tensor, batch_size=2, verbose=1)

    gam_model_2 = IntegratedSaliencyMapping(
        sequential_model,
        'linear',
    )
    gam_model_2.predict(graph_tensor, batch_size=2, verbose=1)

    gam_model_3 = SmoothGradSaliencyMapping(
        sequential_model,
        'linear',
    )
    gam_model_3.predict(graph_tensor, batch_size=2, verbose=1)


@pytest.mark.parametrize('graph_tensor', [graph_tensor])
def test_saliency(graph_tensor):

    graph_tensor = graph_tensor.merge()

    # ragged
    sequential_model = tf.keras.Sequential([
        tf.keras.layers.Input(type_spec=graph_tensor.unspecific_spec),
        GCNConv(128, name='conv_1'),
        GCNConv(128, name='conv_2'),
        GCNConv(128, name='conv_3'),
        GCNConv(128, name='conv_4'),
        Readout(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    gam_model_1 = SaliencyMapping(
        sequential_model,
        'linear',
    )
    gam_model_1(graph_tensor, batch_size=2, verbose=1)

    gam_model_2 = IntegratedSaliencyMapping(
        sequential_model,
        'linear',
    )
    gam_model_2(graph_tensor, batch_size=2, verbose=1)

    gam_model_3 = SmoothGradSaliencyMapping(
        sequential_model,
        'linear',
    )
    gam_model_3(graph_tensor, batch_size=2, verbose=1)


@pytest.mark.parametrize('graph_tensor', [graph_tensor])
def test_saliency_dataset(graph_tensor):

    ds = tf.data.Dataset.from_tensor_slices(graph_tensor).batch(2)

    # ragged
    sequential_model = tf.keras.Sequential([
        tf.keras.layers.Input(type_spec=graph_tensor.unspecific_spec),
        GCNConv(128, name='conv_1'),
        GCNConv(128, name='conv_2'),
        GCNConv(128, name='conv_3'),
        GCNConv(128, name='conv_4'),
        Readout(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    gam_model_1 = SaliencyMapping(
        sequential_model,
        'linear',
    )
    gam_model_1.predict(ds, batch_size=2, verbose=1)

    gam_model_2 = IntegratedSaliencyMapping(
        sequential_model,
        'linear',
    )
    gam_model_2.predict(ds, batch_size=2, verbose=1)

    gam_model_3 = SmoothGradSaliencyMapping(
        sequential_model,
        'linear',
    )
    gam_model_3.predict(ds, batch_size=2, verbose=1)


@pytest.mark.parametrize('graph_tensor', [graph_tensor])
def test_saliency_dataset_merged(graph_tensor):

    ds = tf.data.Dataset.from_tensor_slices(graph_tensor).batch(2).map(
        lambda x: x.merge())

    # ragged
    sequential_model = tf.keras.Sequential([
        tf.keras.layers.Input(type_spec=graph_tensor.merge().unspecific_spec),
        GCNConv(128, name='conv_1'),
        GCNConv(128, name='conv_2'),
        GCNConv(128, name='conv_3'),
        GCNConv(128, name='conv_4'),
        Readout(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    gam_model_1 = SaliencyMapping(
        sequential_model,
        'linear',
    )
    gam_model_1.predict(ds, batch_size=2, verbose=1)

    gam_model_2 = IntegratedSaliencyMapping(
        sequential_model,
        'linear',
    )
    gam_model_2.predict(ds, batch_size=2, verbose=1)

    gam_model_3 = SmoothGradSaliencyMapping(
        sequential_model,
        'linear',
    )
    gam_model_3.predict(ds, batch_size=2, verbose=1)

@pytest.mark.parametrize('graph_tensor', [graph_tensor])
def test_saliency_dataset_with_label(graph_tensor):

    label = tf.constant([1., 2., 3., 4., 5.])
    ds = tf.data.Dataset.from_tensor_slices((graph_tensor, label)).batch(2)

    # ragged
    sequential_model = tf.keras.Sequential([
        tf.keras.layers.Input(type_spec=graph_tensor.unspecific_spec),
        GCNConv(128, name='conv_1'),
        GCNConv(128, name='conv_2'),
        GCNConv(128, name='conv_3'),
        GCNConv(128, name='conv_4'),
        Readout(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    gam_model_1 = SaliencyMapping(
        sequential_model,
        'linear',
    )
    gam_model_1.predict(ds, batch_size=2, verbose=1)

    gam_model_2 = IntegratedSaliencyMapping(
        sequential_model,
        'linear',
    )
    gam_model_2.predict(ds, batch_size=2, verbose=1)

    gam_model_3 = SmoothGradSaliencyMapping(
        sequential_model,
        'linear',
    )
    gam_model_3.predict(ds, batch_size=2, verbose=1)


@pytest.mark.parametrize('graph_tensor', [graph_tensor])
def test_saliency_dataset_with_label_merged(graph_tensor):

    label = tf.constant([1., 2., 3., 4., 5.])
    ds = tf.data.Dataset.from_tensor_slices((graph_tensor, label)).batch(2).map(
        lambda x, y: (x.merge(), y))

    # ragged
    sequential_model = tf.keras.Sequential([
        tf.keras.layers.Input(type_spec=graph_tensor.merge().unspecific_spec),
        GCNConv(128, name='conv_1'),
        GCNConv(128, name='conv_2'),
        GCNConv(128, name='conv_3'),
        GCNConv(128, name='conv_4'),
        Readout(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    gam_model_1 = SaliencyMapping(
        sequential_model,
        'linear',
    )
    gam_model_1.predict(ds, batch_size=2, verbose=1)

    gam_model_2 = IntegratedSaliencyMapping(
        sequential_model,
        'linear',
    )
    gam_model_2.predict(ds, batch_size=2, verbose=1)

    gam_model_3 = SmoothGradSaliencyMapping(
        sequential_model,
        'linear',
    )
    gam_model_3.predict(ds, batch_size=2, verbose=1)


@pytest.mark.parametrize('graph_tensor', [graph_tensor])
def test_saliency_dataset_with_onehot_label(graph_tensor):

    label = tf.constant([
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [1., 0., 0.],
        [0., 1., 0.],
    ])
    ds = tf.data.Dataset.from_tensor_slices((graph_tensor, label)).batch(2).map(
        lambda x, y: (x.merge(), y))

    # ragged
    sequential_model = tf.keras.Sequential([
        tf.keras.layers.Input(type_spec=graph_tensor.merge().unspecific_spec),
        GCNConv(128, name='conv_1'),
        GCNConv(128, name='conv_2'),
        GCNConv(128, name='conv_3'),
        GCNConv(128, name='conv_4'),
        Readout(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(3)
    ])

    gam_model_1 = SaliencyMapping(
        sequential_model,
        'linear',
    )
    gam_model_1.predict(ds, batch_size=2, verbose=1)

    gam_model_2 = IntegratedSaliencyMapping(
        sequential_model,
        'linear',
    )
    gam_model_2.predict(ds, batch_size=2, verbose=1)

    gam_model_3 = SmoothGradSaliencyMapping(
        sequential_model,
        'linear',
    )
    gam_model_3.predict(ds, batch_size=2, verbose=1)



@pytest.mark.parametrize('graph_tensor', [graph_tensor])
def test_gam_ragged(graph_tensor):

    sequential_model = tf.keras.Sequential([
        tf.keras.layers.Input(type_spec=graph_tensor.unspecific_spec),
        GCNConv(128, name='conv_1'),
        GCNConv(128, name='conv_2'),
        Readout(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    gam_model = GradientActivationMapping(sequential_model, ['conv_1', 'conv_2'], 'linear')
    gam_model.predict(graph_tensor, batch_size=2, verbose=1)


@pytest.mark.parametrize('graph_tensor', [graph_tensor])
def test_gam(graph_tensor):

    graph_tensor = graph_tensor.merge()

    sequential_model = tf.keras.Sequential([
        tf.keras.layers.Input(type_spec=graph_tensor.unspecific_spec),
        GCNConv(128, name='conv_1'),
        GCNConv(128, name='conv_2'),
        Readout(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    gam_model = GradientActivationMapping(
        sequential_model, ['conv_1', 'conv_2'], 'linear')
    gam_model(graph_tensor, batch_size=2, verbose=1)


@pytest.mark.parametrize('graph_tensor', [graph_tensor])
def test_gam_dataset(graph_tensor):

    ds = tf.data.Dataset.from_tensor_slices(graph_tensor).batch(2)

    sequential_model = tf.keras.Sequential([
        tf.keras.layers.Input(type_spec=graph_tensor.unspecific_spec),
        GCNConv(128, name='conv_1'),
        GCNConv(128, name='conv_2'),
        Readout(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    gam_model = GradientActivationMapping(
        sequential_model, ['conv_1', 'conv_2'], 'linear')
    gam_model.predict(ds, batch_size=2, verbose=1)


@pytest.mark.parametrize('graph_tensor', [graph_tensor])
def test_gam_dataset_merged(graph_tensor):

    ds = tf.data.Dataset.from_tensor_slices(graph_tensor).batch(2).map(
        lambda x: x.merge())

    sequential_model = tf.keras.Sequential([
        tf.keras.layers.Input(type_spec=graph_tensor.merge().unspecific_spec),
        GCNConv(128, name='conv_1'),
        GCNConv(128, name='conv_2'),
        Readout(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    gam_model = GradientActivationMapping(
        sequential_model, ['conv_1', 'conv_2'], 'linear')
    gam_model.predict(ds, batch_size=2, verbose=1)

@pytest.mark.parametrize('graph_tensor', [graph_tensor])
def test_gam_dataset_with_label(graph_tensor):

    label = tf.constant([1., 2., 3., 4., 5.])
    ds = tf.data.Dataset.from_tensor_slices((graph_tensor, label)).batch(2)

    sequential_model = tf.keras.Sequential([
        tf.keras.layers.Input(type_spec=graph_tensor.unspecific_spec),
        GCNConv(128, name='conv_1'),
        GCNConv(128, name='conv_2'),
        Readout(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    gam_model = GradientActivationMapping(
        sequential_model, ['conv_1', 'conv_2'], 'linear')
    gam_model.predict(ds, batch_size=2, verbose=1)


@pytest.mark.parametrize('graph_tensor', [graph_tensor])
def test_gam_dataset_with_label_merged(graph_tensor):

    label = tf.constant([1., 2., 3., 4., 5.])
    ds = tf.data.Dataset.from_tensor_slices((graph_tensor, label)).batch(2).map(
        lambda x, y: (x.merge(), y))

    # ragged
    sequential_model = tf.keras.Sequential([
        tf.keras.layers.Input(type_spec=graph_tensor.merge().unspecific_spec),
        GCNConv(128, name='conv_1'),
        GCNConv(128, name='conv_2'),
        Readout(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    gam_model = GradientActivationMapping(
        sequential_model, ['conv_1', 'conv_2'], 'linear')
    gam_model.predict(ds, batch_size=2, verbose=1)


@pytest.mark.parametrize('graph_tensor', [graph_tensor])
def test_gam_dataset_with_onehot_label(graph_tensor):

    label = tf.constant([
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [1., 0., 0.],
        [0., 1., 0.],
    ])
    ds = tf.data.Dataset.from_tensor_slices((graph_tensor, label)).batch(2)

    sequential_model = tf.keras.Sequential([
        tf.keras.layers.Input(type_spec=graph_tensor.unspecific_spec),
        GCNConv(128, name='conv_1'),
        GCNConv(128, name='conv_2'),
        Readout(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(3)
    ])

    gam_model = GradientActivationMapping(
        sequential_model, ['conv_1', 'conv_2'], 'linear')
    gam_model.predict(ds, verbose=1)
