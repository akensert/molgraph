import tensorflow as tf
import numpy as np

import os
import math
import json
import multiprocessing

from glob import glob
from time import sleep
from warnings import warn
from contextlib import contextmanager 
from dataclasses import dataclass

from typing import Optional
from typing import Union
from typing import Dict
from typing import List
from typing import Tuple
from typing import Any

from molgraph.tensors.graph_tensor import GraphTensor
from molgraph.chemistry.molecular_encoders import MolecularGraphEncoder
from molgraph.chemistry.molecular_encoders import MolecularGraphEncoder3D
from molgraph.chemistry.benchmark.datasets import Dataset


# TODO: Clean up code.

@dataclass
class Writer:
    path: str
    num_files: Optional[int] = None
    num_processes: Optional[int] = None

    def write(self, data, encoder=None):
        return write(self.path, data, encoder, self.num_files, self.num_processes)
    
@contextmanager
def writer(path, num_files=None, num_processes=None, device='/cpu:0'):
    with tf.device(device):
        yield Writer(path, num_files, num_processes)

def write(
    path: str,
    data: dict,
    encoder: Union[MolecularGraphEncoder, MolecularGraphEncoder3D, None] = None,
    num_files: Optional[int] = None,
    num_processes: Optional[int] = None,
    **kwargs
) -> None:

    '''Writes TF records.

    **Example:**

    >>> x = ['CC', 'CCC', 'CCCC']
    >>> y = [ 5.2,  7.4,   8.1]
    >>> encoder = molgraph.chemistry.MolecularGraphEncoder(
    ...     atom_encoder=molgraph.chemistry.Featurizer([
    ...         molgraph.chemistry.features.Symbol(),
    ...         molgraph.chemistry.features.Hybridization(),
    ...     ])
    ... )
    >>> molgraph.chemistry.tf_records.write( # doctest: +SKIP
    ...     path='/tmp/dummy_records/',
    ...     data={'x': x, 'y': y},
    ...     encoder=encoder
    ... )

    **Important**: If the current enviromnet is running on a GPU by default, 
    please use the `writer` context manager instead:

    >>> with molgraph.chemistry.tf_records.writer('/tmp/dummy_records/') as writer:
    ...    
    ...     # In contrast to previous example, lets obtain the GraphTensor 
    ...     # instances outside the write function. Note: this would cause 
    ...     # issues if run on a GPU (namely, without the writer context manager)
    ...
    ...     t1 = GraphTensor(
    ...         node_feature=tf.constant([[1.]]), 
    ...         edge_src=tf.constant([], dtype=tf.int64),
    ...         edge_dst=tf.constant([], dtype=tf.int64))
    ...
    ...     x2 = GraphTensor(
    ...         node_feature=tf.constant([[1.], [2.]]), 
    ...         edge_src=tf.constant([0, 1], dtype=tf.int64),
    ...         edge_dst=tf.constant([1, 0], dtype=tf.int64))
    ...
    ...     x3 = GraphTensor(
    ...         node_feature=tf.constant([[1.], [2.], [3.]]), 
    ...         edge_src=tf.constant([0, 1, 2], dtype=tf.int64),
    ...         edge_dst=tf.constant([1, 2, 0], dtype=tf.int64))
    ...     
    ...     # Should not specify path, num_files or num_processes; encoder is 
    ...     # optional: here not needed as we already obtained the graph tensors.
    ...     writer.write( # doctest: +SKIP
    ...         data={'x': [x1, x2, x3], 'y': [0., 1., 2.]},
    ...         encoder=None # encoder not needed as graph tensors are passed
    ...     )
    ...
    >>> # load tf records as tf.data.Dataset
    >>> ds = molgraph.chemistry.tf_records.load( # doctest: +SKIP
    ...     '/tmp/dummy_records/'
    ... )

    Args:
        path (str):
            The path to write TF records to (save path). Should not include
            file name. File names are automatically determined.
        data (dict):
            The data to be written as TF records. The keys of the data (dict),
            are the name of the data fields, while the values are the actual
            values (of the fields). E.g., 
            ``{'x': ['CC', 'CCO'], 'y': [4.1, 2.4]}``. The ``encoder`` will be 
            applied to the mandatory ``data['x']`` field.
        encoder (MolecularGraphEncoder, MolecularGraphEncoder3D, None):
            The encoder to be applied to ``data['x']``. The encoder transforms
            the string (or rdkit.Chem.Mol) representations of molecules into
            a ``GraphTensor``. If None, it is assumed that ``data['x']`` already
            contains GraphTensor instances in a list: [gt_1, gt_2, ..., gt_n]. 
            Default to None.
        num_files (int, None):
            The number of TF record files to write to. If None, num_files will
            be set to ``num_processes``. Default to None.
        num_processes (int, None):
            The number of worker processes to use. If None, 
            ``multiprocessing.cpu_count()`` will be used. Using multiple worker 
            processes significantly speeds up writing of TF records. If 
            ``num_files`` < ``num_processes``, only ``num_files`` processes 
            will be used. Default to None.

    Returns:
        ``None``
    '''
    
        
    inputs = kwargs.pop('inputs', None)
    if inputs is not None:
        warn(
            (
                '`inputs` argument will be depracated in the near future, '
                ' please use `data` instead.'
            ),
            DeprecationWarning,
            stacklevel=2
        )

    device = kwargs.pop('device', None)
    if device is not None:
        warn(
            (
                '`device` argument will be depracated in the near future, '
                ' please use tf_records.writer instead.'
            ),
            DeprecationWarning,
            stacklevel=2
        )

    assert 'x' in data, ('`data` requires field `x`.')

    os.makedirs(path, exist_ok=True)

    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    if num_files is None:
        num_files = num_processes

    chunk_size = math.ceil(len(data['x']) / num_files)

    spec = {}
    # Obtain spec for each data component, and also chunk the data components.
    for key, value in data.items():

        if key == 'x':
            graph_tensor = (
                value[0] if isinstance(value[0], GraphTensor) else 
                encoder(value[0])
            )
            graph_tensor_spec = tf.type_spec_from_value(graph_tensor)
            # graph_tensor_spec.pop('graph_indicator')
            def add_batch_dim(x):
                return tf.TensorSpec(shape=[None] + x.shape[1:], dtype=x.dtype)
            
            data_spec = graph_tensor_spec.data_spec 
            data_spec = tf.nest.map_structure(add_batch_dim, data_spec)
            spec[key] = GraphTensor.Spec(sizes=graph_tensor_spec.sizes, **data_spec)
        else:
            spec[key] = tf.type_spec_from_value(value[0])

        data[key] = [
            value[i * chunk_size: (i + 1) * chunk_size]
            for i in range(num_files)
        ]

    # Save specs to json file, in the same folder as the tf records.
    _specs_to_json(spec, os.path.join(path, 'spec.json'))
    
    # Create different paths for associated with each chunk of data.
    paths = [
        os.path.join(path, f'tfrecord-{i:04d}.tfrecord')
        for i in range(num_files)
    ]
    
    # data contain nested chunks: 
    # {'x': [x_chunk_1, x_chunk_2, ...], 'y': [y_chunk_1, y_chunk_2, ...]}
    data_keys = data.keys()
    data_values = data.values()
    
    processes = []
    # Loop chunk-wise:
    # [path_1, x_chunk_1, y_chunk_1], ..., [path_n, x_chunk_n, y_chunk_n]
    
    for path, *values in zip(paths, *data_values):

        # Do not start new processes if 'num_processes' are still alive
        while len(processes) >= num_processes:
            for process in processes:
                if not process.is_alive():
                    processes.remove(process)
            else:
                sleep(0.1)
                continue

        # Create dictionary to keep track of the different data components:
        # {'x': x_chunk_i, 'y': y_chunk_i}
        data_chunk = dict(zip(data_keys, values))
        
        # Start process. Will write (nested) data chunk to tf records
        process = multiprocessing.Process(
            target=_write_tfrecords_to_file,
            args=(path, data_chunk, encoder)
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

def load(
    path: str,
    extract_tuple: Optional[Union[List[str], Tuple[str]]] = None,
    shuffle_tf_records: bool = False,
) -> tf.data.Dataset:
    '''Loads TF records.

    **Example:**

    >>> ds = molgraph.chemistry.tf_records.load( # doctest: +SKIP
    ...     path='/tmp/dummy_records/', # extract_tuple=('x', 'y')
    ... )
    >>> ds = ds.shuffle(3) # doctest: +SKIP
    >>> ds = ds.batch(2) # doctest: +SKIP
    >>> ds = ds.prefetch(-1) # doctest: +SKIP
    >>> for batch in ds.take(1): # doctest: +SKIP
    ...     print(batch['x'])

    Args:
        path (str):
            Path to TF record files (excluding file names).
        extract_tuple (list[str], tuple[str], None):
            Optionally specify what fields to extract. If None, returned TF
            dataset will produce dictionaries (corresponding to ``inputs``
            passed to ``write``). If not None, tuples will be produced.
            Default to None.
        shuffle_tf_records (bool):
            Whether tf record files should be shuffled. Default to False.
            Recommended to be set to True when loading training dataet.

    Returns:
        ``tf.data.Dataset``: A TF dataset ready to be passed to GNN models.
    '''
    specs = _specs_from_json(os.path.join(path, 'spec.json'))
    filenames = glob(os.path.join(path, '*.tfrecord'))
    filenames = sorted(filenames)
    num_files = len(filenames)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if shuffle_tf_records:
        dataset = dataset.shuffle(num_files)
    dataset = dataset.interleave(
        tf.data.TFRecordDataset, num_parallel_calls=1)
    dataset = dataset.map(
        lambda data: _parse_features(
            data, 
            specs=specs, 
            extract_tuple=extract_tuple
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset

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

def _write_tfrecords_to_file(
    path: str,
    data_chunk: Dict[str, Any],
    encoder: Union[
        MolecularGraphEncoder, 
        MolecularGraphEncoder3D, 
        None
    ] = None,
) -> None:
    # Zip nested data chunks:
    # {'x': ['C', 'CC', ...], 'y': [1, 2, ...]} -> [('C', 1), ('CC', 2), ...]
    data_keys = data_chunk.keys()
    data_values = list(zip(*data_chunk.values()))

    # Write tf records using 'device'
    with tf.io.TFRecordWriter(path) as writer:

        # Loop over each tuple in chunk (e.g. each (x, y) pair)
        for value in data_values:
    
            # Create dictionary to keep track of data component:
            # ('C', 1) -> {'x': 'C', 'y': 1}
            value = dict(zip(data_keys, value))
            x = value.pop('x')
           
            if not isinstance(x, GraphTensor):
                graph_tensor = encoder(x)
                if graph_tensor is None:
                    # TODO: Raise warning?
                    continue
            else:
                graph_tensor = x

            # Extract graph tensor data components and convert to bytes feature
            graph_tensor_data = graph_tensor.data

            # graph_tensor_data.pop('graph_indicator')
            example = tf.nest.map_structure(
                lambda x: _to_bytes_feature(x), graph_tensor_data)

            # Convert remaining data components (non-GraphTensor instances)
            # to bytes features
            for k, v in value.items():
                example[k] = _to_bytes_feature(v)

            # Serialize and write bytes features to tf records.
            serialized = _serialize_example(example)

            writer.write(serialized)

def _parse_features(
    example_proto: tf.Tensor,
    specs: Dict[str, Union[GraphTensor.Spec, tf.TensorSpec]],
    extract_tuple: Optional[Union[List[str], Tuple[str]]] = None,
) -> Dict[str, Union[GraphTensor, tf.Tensor]]:

    graph_tensor_spec = specs.pop('x')
    graph_tensor_data_spec = graph_tensor_spec.data_spec

    feature_description = tf.nest.map_structure(
        lambda _: tf.io.FixedLenFeature([], tf.string), 
        graph_tensor_data_spec)
    
    example = tf.io.parse_single_example(
        serialized=example_proto, 
        features=feature_description)
    
    graph_tensor_data = tf.nest.map_structure(
        lambda x, s: tf.io.parse_tensor(x, s.dtype), 
        example, 
        graph_tensor_data_spec)
    
    graph_tensor_data = tf.nest.map_structure(
        lambda x, s: tf.ensure_shape(x, s.shape),
        graph_tensor_data,
        graph_tensor_data_spec
    )

    graph_tensor = GraphTensor(**graph_tensor_data)

    if specs:
        feature_description = tf.nest.map_structure(
            lambda _: tf.io.FixedLenFeature([], tf.string), specs)
        example = tf.io.parse_single_example(example_proto, feature_description)
        data = tf.nest.map_structure(
            lambda x, s: tf.ensure_shape(
                tf.io.parse_tensor(x, s.dtype), s.shape), example, specs)
        data['x'] = graph_tensor
    else:
        data = {'x': graph_tensor}

    if extract_tuple is not None:
        data = tuple(data[key] for key in extract_tuple if key in data)

    return data

def _serialize_example(feature: Dict[str, tf.train.Feature]) -> bytes:
    'Converts a dictionary of (str, feature) pairs to a bytes string.'
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def _to_bytes_feature(value: Union[tf.Tensor, np.ndarray]) -> tf.train.Feature:
    'Encodes array as a bytes feature.'
    value = tf.io.serialize_tensor(value)
    value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _serialize_spec(
    spec: Union[GraphTensor.Spec, tf.TensorSpec]
) -> Union[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    if isinstance(spec, GraphTensor.Spec):
        return {
            k: {'shape': v.shape.as_list(), 'dtype': v.dtype.name}
            for (k, v) in spec.data_spec.items()}
    return {
        'shape': spec.shape.as_list(), 'dtype': spec.dtype.name}

def _deserialize_spec(
    serialized_spec: Union[Dict[str, Dict[str, Any]], Dict[str, Any]]
) -> Union[GraphTensor.Spec, tf.TensorSpec]:
    if 'node_feature' in serialized_spec:
        return GraphTensor.Spec(**{
            k: _deserialize_spec(v) for (k, v) in serialized_spec.items()
        })
    return tf.TensorSpec(
        shape=serialized_spec['shape'], dtype=serialized_spec['dtype'])

def _specs_to_json(
    specs: Dict[str, Union[GraphTensor.Spec, tf.TensorSpec]],
    path: str
) -> None:
    specs = {k: _serialize_spec(v) for (k, v) in specs.items()}
    with open(path, 'w') as out_file:
        json.dump(specs, out_file)

def _specs_from_json(
    path: str
) -> Dict[str, Union[GraphTensor.Spec, tf.TensorSpec]]:
    with open(path) as in_file:
        specs = json.load(in_file)
    return {k: _deserialize_spec(v) for (k, v) in specs.items()}