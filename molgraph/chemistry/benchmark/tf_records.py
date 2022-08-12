import math
import os
import multiprocessing
from tqdm import tqdm
from functools import partial
import json
from glob import glob
import tensorflow as tf
import numpy as np

from rdkit import Chem
from typing import Optional
from typing import Union
from typing import Dict
from typing import List
from typing import Tuple
from typing import Sequence
from typing import Any

from molgraph.tensors import GraphTensor
from molgraph.tensors import GraphTensorSpec
from molgraph.chemistry import MolecularGraphEncoder
from molgraph.chemistry import MolecularGraphEncoder3D
from molgraph.chemistry.benchmark.datasets import Dataset


def _serialize_example(feature: Dict[str, tf.train.Feature]) -> bytes:
    """Converts a dictionary of str:feature pairs to a bytes string."""
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def _to_bytes_feature(value: Union[tf.Tensor, np.ndarray]) -> tf.train.Feature:
    """Encodes array as a bytes feature."""
    value = tf.io.serialize_tensor(value)
    value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _serialize_spec(
    spec: Union[GraphTensorSpec, tf.TensorSpec]
) -> Union[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    if isinstance(spec, GraphTensorSpec):
        return {
            k: {'shape': v.shape.as_list(), 'dtype': v.dtype.name}
            for (k, v) in spec._data_spec.items()}
    return {
        'shape': spec.shape.as_list(), 'dtype': spec.dtype.name}

def _deserialize_spec(
    serialized_spec: Union[Dict[str, Dict[str, Any]], Dict[str, Any]]
) -> Union[GraphTensorSpec, tf.TensorSpec]:
    if 'node_feature' in serialized_spec:
        return GraphTensorSpec({
            k: _deserialize_spec(v) for (k, v) in serialized_spec.items()
        })
    return tf.TensorSpec(
        shape=serialized_spec['shape'], dtype=serialized_spec['dtype'])

def _specs_to_json(
    specs: Dict[str, Union[GraphTensorSpec, tf.TensorSpec]],
    path: str
) -> None:
    specs = {k: _serialize_spec(v) for (k, v) in specs.items()}
    with open(path, 'w') as out_file:
        json.dump(specs, out_file)

def _specs_from_json(
    path: str
) -> Dict[str, Union[GraphTensorSpec, tf.TensorSpec]]:
    with open(path) as in_file:
        specs = json.load(in_file)
    return {k: _deserialize_spec(v) for (k, v) in specs.items()}

def _write_tfrecords_to_file(
    inputs: List[Any],
    input_keys: List[str],
    encoder: Union[MolecularGraphEncoder, MolecularGraphEncoder3D],
    device: str = '/cpu:0',
) -> None:
    """Writes data to a tf record file. This function is called from
    `write_tfrecords`.
    """
    path, *inputs = inputs
    inputs = list(zip(*inputs))
    with tf.device(device), tf.io.TFRecordWriter(path) as writer:
        for inp in inputs:
            inp = dict(zip(input_keys, inp))
            x = inp.pop('x')
            tensor = encoder(x)
            if tensor is None:
                continue
            data = tensor._data.copy()
            data.pop('graph_indicator')
            example = tf.nest.map_structure(
                lambda x: _to_bytes_feature(x), data)
            for k, v in inp.items():
                example[k] = _to_bytes_feature(v)
            serialized = _serialize_example(example)
            writer.write(serialized)

def write(
    path: str,
    inputs: dict,
    encoder: Union[MolecularGraphEncoder, MolecularGraphEncoder3D],
    num_files: Optional[int] = None,
    processes: Optional[int] = None,
    device: str = '/cpu:0'
) -> None:

    if processes is None:
        processes = multiprocessing.cpu_count()

    if num_files is None:
        num_files = processes

    chunk_size = math.ceil(len(inputs['x']) / num_files)

    spec = {}
    for key, value in inputs.items():
        if key == 'x':
            graph_tensor_spec = encoder(value[0]).unspecific_spec._data_spec
            graph_tensor_spec.pop('graph_indicator')
            spec[key] = GraphTensorSpec(graph_tensor_spec)
        else:
            # tf.convert_to_tensor?
            spec[key] = tf.type_spec_from_value(value[0])

        inputs[key] = [
            value[i * chunk_size: (i + 1) * chunk_size]
            for i in range(num_files)
        ]

    input_keys = list(inputs.keys())
    inputs = list(inputs.values())

    directory = os.path.dirname(path)

    os.makedirs(directory, exist_ok=True)

    _specs_to_json(spec, os.path.join(directory, 'spec.json'))

    path = [
        os.path.join(directory, f'tfrecord-{i:04d}.tfrecord')
        for i in range(num_files)
    ]

    inputs.insert(0, path)

    with multiprocessing.Pool(processes) as pool:
        pool.map(
            func=partial(
                _write_tfrecords_to_file,
                input_keys=input_keys,
                encoder=encoder,
                device=device
            ),
            iterable=zip(*inputs)
        )

def write_from_dataset(
    path: str,
    dataset: Dataset,
    encoder: MolecularGraphEncoder,
    num_files: Optional[int] = None,
    processes: Optional[int] = None,
    device: str = '/cpu:0'
) -> None:
    for key, value in dataset.items():
        path_subset = os.path.join(path, key, '')
        write(path_subset, value, encoder, num_files, processes, device)

def _parse_features(
    example_proto: tf.Tensor,
    specs: Dict[str, Union[GraphTensorSpec, tf.TensorSpec]],
    extract_tuple: Optional[Union[List[str], Tuple[str]]] = None,
) -> Dict[str, Union[GraphTensor, tf.Tensor]]:

    # parse graph tensor
    x_spec = specs.pop('x')

    feature_description = tf.nest.map_structure(
        lambda _: tf.io.FixedLenFeature([], tf.string), x_spec._data_spec)

    example = tf.io.parse_single_example(example_proto, feature_description)

    x = tf.nest.map_structure(
        lambda x, s: tf.io.parse_tensor(x, s.dtype), example, x_spec._data_spec)

    feature_description = tf.nest.map_structure(
        lambda _: tf.io.FixedLenFeature([], tf.string), specs)

    example = tf.io.parse_single_example(example_proto, feature_description)

    data = tf.nest.map_structure(
        lambda x, s: tf.ensure_shape(
            tf.io.parse_tensor(x, s.dtype), s.shape), example, specs)

    x_spec = tf.nest.map_structure(
        lambda spec: tf.RaggedTensorSpec(
            spec.shape, spec.dtype, 0, x['edge_dst'].dtype),
        x_spec._data_spec)
    data['x'] = GraphTensor(x, x_spec)
    if extract_tuple is not None:
        data = tuple(data[key] for key in extract_tuple if key in data)
    return data

def load(
    path: str,
    extract_tuple: Optional[Union[List[str], Tuple[str]]] = None,
    shuffle_tfrecords: bool = True,
) -> tf.data.Dataset:
    specs = _specs_from_json(os.path.join(path, 'spec.json'))
    filenames = glob(os.path.join(path, '*.tfrecord'))
    num_files = len(filenames)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if shuffle_tfrecords:
        dataset = dataset.shuffle(num_files)
    dataset = dataset.interleave(
        tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(
        partial(_parse_features, specs=specs, extract_tuple=extract_tuple),
        num_parallel_calls=tf.data.AUTOTUNE)
    return dataset
