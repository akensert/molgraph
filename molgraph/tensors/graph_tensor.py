import tensorflow as tf
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import type_spec
from tensorflow.experimental import ExtensionType
from tensorflow.experimental import BatchableExtensionType
import numpy as np

from typing import Optional
from typing import Mapping
from typing import List
from typing import Tuple
from typing import Union
from typing import Any
from typing import Type


_allowable_input_types = (
    tf.Tensor,
    tf.RaggedTensor,
    np.ndarray,
    list,
    tuple
)
_required_fields = ['edge_dst', 'edge_src', 'node_feature']

_non_updatable_fields = ['edge_dst', 'edge_src', 'graph_indicator']

NestedTensors = Mapping[str, Union[tf.Tensor, tf.RaggedTensor]]
NestedTensorSpecs = Mapping[str, Union[tf.TensorSpec, tf.RaggedTensorSpec]]
NestedArrays = Mapping[
    str, Union[ # TensorLike
        tf.Tensor,
        tf.RaggedTensor,
        np.ndarray,
        list,
        tuple
    ]
]



class GraphTensor(composite_tensor.CompositeTensor):

    '''A composite tensor encoding a molecular graph.

    The molecular graph (GraphTensor) could encode a single subgraph (single
    molecule) or multiple subgraphs (multiple molecules). The GraphTensor
    can either encode the molecular graph as a single (disjoint) graph (nested
    tensors) or as multiple subgraphs (nested ragged tensors). The former is
    advantageous for efficient computation while the latter is advantageous
    for batching (via e.g., the tf._data.Dataset API).
    '''

    __slots__ = ['_data', '_spec']

    def __init__(
        self,
        data: Optional[NestedArrays] = None,
        spec: Optional[NestedTensorSpecs] = None,
        **data_kwargs # allows to pass data as keyword arguments
    ) -> None:
        'Initializes attributes of graph tensor'
        super().__init__()

        if data is None:
            data = {}

        data.update(data_kwargs)
        data = _maybe_convert_to_tensors(
            data, check_keys=True, check_values=True)

        if spec is None:
            spec = tf.nest.map_structure(tf.type_spec_from_value, data)
        else:
            tf.nest.assert_same_structure(data, spec)

        data, spec = _maybe_add_graph_indicator(data, spec)

        self._spec = GraphTensorSpec(spec)
        self._data = data

    def update(self, new_data: NestedArrays) -> 'GraphTensor':
        '''Updates existing data fields or adds new data fields.

        Constraints are put on the update method: new data needs to
        match the size of existing data (e.g., same number of nodes or edges).
        '''

        new_data = _maybe_convert_to_tensors(new_data, check_values=True)

        data = self._data.copy()

        def convert_tensor(
            new_value: Union[tf.Tensor, tf.RaggedTensor],
            old_value: Union[tf.Tensor, tf.RaggedTensor],
        ) -> Union[tf.Tensor, tf.RaggedTensor]:
            if (
                isinstance(new_value, tf.RaggedTensor) and
                isinstance(old_value, tf.Tensor)
            ):
                new_value = new_value.flat_values
            elif (
                isinstance(new_value, tf.Tensor) and
                isinstance(old_value, tf.RaggedTensor)
            ):
                new_value = old_value.with_flat_values(new_value)
            else:
                new_value = new_value

            return new_value

        for k in list(new_data.keys()):

            if k in _non_updatable_fields:
                raise ValueError(f'{k} cannot be updated.')

            new_value = new_data.pop(k)

            # Make sure the new value has the same number of nodes or edges
            # as the exisiting values of the graph tensor.
            _assert_compatible_sizes(
                new_value, data['node_feature'], data['edge_dst'])

            if k in data:
                data[k] = convert_tensor(
                    new_value, data[k])
            elif _compatible_sizes(new_value, data['node_feature']):
                data[k] = convert_tensor(new_value, data['node_feature'])
            else:
                data[k] = convert_tensor(new_value, data['edge_dst'])

        return self.__class__(data)

    def remove(
        self,
        fields: Union[str, List[str]]
    ) -> 'GraphTensor':
        'Removes data fields'
        data = self._data.copy()
        if isinstance(fields, str):
            fields = [fields]
        for field in fields:
            if field in _non_updatable_fields:
                raise ValueError(f'{k} cannot be removed.')
            data.pop(field)
        return self.__class__(data)

    def merge(self) -> 'GraphTensor':
        'Merges subgraphs into a single disjoint graph'

        _assert_mergeable(self._data)

        data = self._data.copy()

        increment = data['node_feature'].row_starts()
        indices = data['edge_dst'].value_rowids()
        graph_indicator = data['node_feature'].value_rowids()
        increment = tf.cast(increment, dtype=data['edge_dst'].dtype)
        data = tf.nest.map_structure(lambda x: x.flat_values, data)
        data['edge_dst'] += tf.gather(increment, indices)
        data['edge_src'] += tf.gather(increment, indices)
        data['graph_indicator'] = graph_indicator

        return self.__class__(data)

    def separate(self) -> 'GraphTensor':
        'Separates the (single disjoint) graph into its subgraphs'

        _assert_separable(self._data)

        data = self._data.copy()

        graph_indicator = data.pop('graph_indicator')
        edge_dst = data.pop('edge_dst')
        edge_src = data.pop('edge_src')

        graph_indicator_edges = tf.gather(graph_indicator, edge_dst)

        def to_ragged_tensor(
            tensor: Union[tf.Tensor, tf.RaggedTensor],
            num_subgraphs: tf.Tensor,
        ) -> tf.RaggedTensor:

            if isinstance(tensor, tf.RaggedTensor):
                return tensor

            value_rowids, nrows = tf.cond(
                _compatible_sizes(tensor, graph_indicator),
                lambda: (graph_indicator, num_subgraphs),
                lambda: tf.cond(
                    _compatible_sizes(tensor, graph_indicator_edges),
                    lambda: (graph_indicator_edges, num_subgraphs),
                    lambda: (tf.zeros(
                        tf.shape(tensor)[0], dtype=graph_indicator.dtype),
                            tf.constant(1, dtype=num_subgraphs.dtype))
                )
            )
            return tf.RaggedTensor.from_value_rowids(
                tensor, value_rowids, nrows)

        num_subgraphs = self.num_subgraphs
        data = tf.nest.map_structure(
            lambda x: to_ragged_tensor(x, num_subgraphs), data)
        decrement = tf.gather(
            data['node_feature'].row_starts(), graph_indicator_edges)
        decrement = tf.cast(decrement, dtype=edge_dst.dtype)
        data['edge_dst'] = tf.RaggedTensor.from_value_rowids(
            edge_dst - decrement, graph_indicator_edges, num_subgraphs)
        data['edge_src'] = tf.RaggedTensor.from_value_rowids(
            edge_src - decrement, graph_indicator_edges, num_subgraphs)

        return self.__class__(data)

    @property
    def shape(self) -> tf.TensorShape:
        'Partial shape of graph tensor (based on `node_feature`)'
        return self._spec.shape

    @property
    def dtype(self) -> tf.DType:
        'Partial dtype of graph tensor (based on `node_feature`)'
        return self._spec.dtype

    @property
    def rank(self) -> int:
        'Partial rank of graph tensor (based on `node_feature`)'
        return self._spec.rank

    @property
    def spec(self) -> 'GraphTensorSpec':
        'CompositeTensor API.'
        return self._type_spec

    @property
    def _type_spec(self) -> 'GraphTensorSpec':
        'CompositeTensor API.'
        return self._spec

    @property
    def unspecific_spec(self):
        '''Unspecific spec of graph tensor.

        Specifically, the shape/size of the first dimension is set to None. This is
        necessary when loading and using saved models (as number of nodes and edges vary).
        '''
        def modify_spec(x):
            if isinstance(x, tf.RaggedTensorSpec):
                return tf.RaggedTensorSpec(
                    shape=tf.TensorShape([None]).concatenate(x.shape[1:]),
                    dtype=x.dtype,
                    row_splits_dtype=x.row_splits_dtype,
                    ragged_rank=x.ragged_rank,
                    flat_values_spec=x.flat_values_spec)
            return tf.TensorSpec(
                shape=tf.TensorShape([None]).concatenate(x.shape[1:]),
                dtype=x.dtype)

        return GraphTensorSpec(
            tf.nest.map_structure(modify_spec, self._spec._data_spec))

    @property
    def num_subgraphs(self):
        if 'graph_indicator' in self._data:
            return tf.math.reduce_max(self._data['graph_indicator']) + 1
        return self._data['node_feature'].nrows()

    def __getattr__(self, name: str) -> Union[tf.Tensor, tf.RaggedTensor, Any]:
        'Access different data fields as attributes ".name_of_attribute'
        if name in object.__getattribute__(self, '_data'):
            return self._data[name]
        return object.__getattribute__(self, name)

    def __getitem__(
        self,
        index: Union[str, List[int]]
    ) -> Union[tf.RaggedTensor, tf.Tensor, 'GraphTensor']:
        'Access data fields or subgraphs via indexing "[]"'
        if isinstance(index, str):
            return self._data[index]
        if isinstance(index, slice):
            index = _slice_to_tensor(index, self.num_subgraphs)
        return tf.gather(self, index)

    def __iter__(self):
        'Allows graph tensor to be iterated'
        if not tf.executing_eagerly():
            raise ValueError(
                'Can only iterate over `GraphTensor` in eager mode.')
        return _Iterator(self, limit=self.num_subgraphs)

    def __repr__(self) -> str:
        'Representation of the graph tensor.'
        fields = []
        for key, value in self._spec._data_spec.items():
            if isinstance(self._data[key], tf.RaggedTensor):
                fields.append(
                    "{}=<tf.RaggedTensor: ".format(key) +
                    "shape={}, ".format(value.shape) +
                    "dtype={}, ".format(value.dtype.name) +
                    "ragged_rank={}, ".format(value.ragged_rank) +
                    "row_splits_dtype={}>".format(value.row_splits_dtype.name)
                )
            elif isinstance(self._data[key], tf.Tensor):
                fields.append(
                    "{}=<tf.Tensor: ".format(key) +
                    "shape={}, ".format(value.shape) +
                    "dtype={}>".format(value.dtype.name)
                )
            else:
                fields.append("{}=<unknown>".format(key))
        return f"{self.__class__.__name__}({', '.join(fields)})"


@type_spec.register('molgraph.tensors.graph_tensor.GraphTensorSpec')
class GraphTensorSpec(type_spec.BatchableTypeSpec):

    __slots__ = ['_data_spec', '_shape', '_dtype']

    def __init__(
        self,
        data_spec: NestedTensorSpecs,
        shape: Optional[tf.TensorShape] = None,
        dtype: Optional[tf.DType] = None
    ) -> None:
        'Initializes attributes of graph tensor spec'
        super().__init__()

        self._data_spec = data_spec

        if shape is None or dtype is None:
            feature_spec = self._data_spec['node_feature']
            if shape is None:
                shape = tf.TensorShape(feature_spec.shape)
            if dtype is None:
                dtype = feature_spec.dtype

        self._shape = shape
        self._dtype = dtype

    @property
    def value_type(self) -> Type[GraphTensor]:
        'ExtensionType API'
        return GraphTensor

    def with_shape(self, shape: tf.TensorShape) -> 'GraphTensorSpec':
        'Keras API'
        return self.__class__(self._data_spec, shape, self._dtype)

    @property
    def shape(self) -> tf.TensorShape:
        'Partial shape of spec (based on `node_feature`).'
        return self._shape

    @property
    def dtype(self) -> tf.DType:
        'Partial dtype of spec (based on `node_feature`).'
        return self._dtype

    @property
    def rank(self) -> int:
        'Partial rank of spec (based on `node_feature`).'
        return self._shape.rank

    @classmethod
    def from_value(cls, value: GraphTensor) -> 'GraphTensorSpec':
        'ExtensionType API'
        return value._type_spec

    def _serialize(
        self
    ) -> Tuple[NestedTensorSpecs, tf.TensorShape, tf.DType]:
        'ExtensionType API'
        return (self._data_spec, self._shape, self._dtype)

    @classmethod
    def _deserialize(
        cls,
        serialization: Tuple[
            NestedTensorSpecs,
            Union[tf.TensorShape, None],
            Union[tf.DType, None]
        ]
    ) -> 'GraphTensorSpec':
        'ExtensionType API'
        data_spec, shape, dtype = serialization
        return cls(data_spec, shape, dtype)

    @property
    def _component_specs(self) -> NestedTensorSpecs:
        'ExtensionType API'
        return self._data_spec

    def _to_components(self, value: GraphTensor) -> NestedTensors:
        'ExtensionType API'
        return value._data.copy()

    def _from_components(self, components: NestedTensors) -> GraphTensor:
        'ExtensionType API'
        return self.value_type(components)

    def _batch(self, batch_size: Union[int, None]) -> 'GraphTensorSpec':
        'BatchableExtensionType API'
        batched_data_spec = tf.nest.map_structure(
            lambda spec: spec._batch(batch_size), self._data_spec)
        shape = tf.TensorShape([batch_size]).concatenate(self._shape)
        return self.__class__(batched_data_spec, shape, self._dtype)

    def _unbatch(self) -> 'GraphTensorSpec':
        'BatchableExtensionType API'
        unbatched_data_spec = tf.nest.map_structure(
            lambda spec: spec._unbatch(), self._data_spec)
        shape = self._shape[1:]
        return self.__class__(unbatched_data_spec, shape, self._dtype)

    def _to_legacy_output_types(self):
        'Legacy method of ExtensionType API'
        return self._dtype

    def _to_legacy_output_shapes(self):
        'Legacy method of ExtensionType API'
        return self._shape

    def _to_legacy_output_classes(self):
        'Legacy method of ExtensionType API'
        return self


def _maybe_convert_to_tensors(
    data: NestedArrays,
    check_values: bool = False,
    check_keys: bool = False,
) -> NestedTensors:
    'Converts data values to tensors'

    if check_values:
        _check_data_values(data)

    if check_keys:
        _check_data_keys(data)

    def _is_rectangular(x):
        'Checks if tensor is rectangular (non-ragged)'
        lengths = set()
        for xi in x:
            if not isinstance(xi, (np.ndarray, list, tuple)):
                lengths.add(0)
            else:
                lengths.add(len(xi))
        return len(lengths) <= 1

    def maybe_convert(x):
        'Convert to tensor or ragged tensor if needed'
        if tf.is_tensor(x):
            if isinstance(x, tf.Tensor):
                return x
            elif isinstance(x, tf.RaggedTensor):
                _check_ragged_rank(x)
                return x
            else:
                raise ValueError(
                    'Tensor needs to be either `tf.Tensor` or `tf.RaggedTensor`.')
        if _is_rectangular(x):
            return tf.convert_to_tensor(x)
        return tf.ragged.constant(x, ragged_rank=1) # Pretty slow

    data = {k: maybe_convert(v) for (k, v) in data.items()}
    _check_tensor_types(data)
    return data

def _check_data_keys(data: NestedArrays):
    'Asserts that necessary fields exist in the graph (tensor)'
    return [
        tf.Assert(
            req_field in data, [f'`data` requires `{req_field}` field']
        )
        for req_field in _required_fields]

def _check_data_values(data: NestedArrays):
    'Asserts that the values of the data fields are of correct type'
    return [
        tf.Assert(
            isinstance(v, _allowable_input_types),
            [
                f'Field `{k}` is needs to be a `tf.Tensor`, ' +
                '`tf.RaggedTensor`, `np.ndarray`, `list` or `tuple`'
            ]
        )
        for (k, v) in data.items()]

def _check_tensor_types(data: NestedTensors):
    'Asserts that all nested values of data are of the same tensor type'
    tests = [isinstance(x, tf.Tensor) for x in data.values()]
    same_types = all(tests) or not any(tests)
    return tf.Assert(
        same_types, [
            f'Nested tensors are not the same type. ' +
             'Found both `tf.Tensor`s and `tf.RaggedTensor`s'])

def _check_ragged_rank(x: tf.RaggedTensor):
    return tf.Assert(
        tf.math.equal(x.ragged_rank, 1),
        ['Ragged rank of component needs to be 1.']
    )

def _compatible_sizes(
    a: Union[tf.Tensor, tf.RaggedTensor],
    b: Union[tf.Tensor, tf.RaggedTensor],
) -> bool:
    'Checks if the two inputs have the same number of nodes or edges'
    def _get_size(x):
        'Get number of nodes or edges'
        if isinstance(x, tf.RaggedTensor):
            x = x.flat_values
        return tf.shape(x)[0]
    return _get_size(a) == _get_size(b)

def _assert_compatible_sizes(
    target: Union[tf.Tensor, tf.RaggedTensor],
    *comparators: Union[tf.Tensor, tf.RaggedTensor]
):
    return tf.Assert(
        tf.math.reduce_any(
            tf.nest.map_structure(
                lambda comparator: _compatible_sizes(target, comparator),
                comparators
            )
        ), [
            'At least one of the added fields does not match ' +
            'the size of the existing fields. Namely, one of the ' +
            'added fields did not have the same numeber of ' +
            'nodes or edges as the existig fields'
        ]
    )

def _maybe_add_graph_indicator(
    data: NestedTensors,
    spec: NestedTensorSpecs,
) -> NestedTensors:
    'Maybe adds `graph_indicator` to data and spec.'
    if (
        'graph_indicator' not in data and
        isinstance(data['node_feature'], tf.Tensor)
    ):
        data['graph_indicator'] = tf.zeros(
            tf.shape(data['node_feature'])[0],
            dtype=data['edge_dst'].dtype)
        spec['graph_indicator'] = tf.type_spec_from_value(
            data['graph_indicator'])
        if isinstance(spec['edge_dst'], tf.RaggedTensorSpec):
            x = spec['graph_indicator']
            spec['graph_indicator'] = tf.RaggedTensorSpec(
                x.shape, x.dtype, spec['edge_dst'].ragged_rank, spec['edge_dst'].row_splits_dtype
            )
    return data, spec

def _assert_mergeable(data: NestedTensors):
    'Asserts that all nested tensors are ragged.'
    return tf.Assert(
        all([isinstance(x, tf.RaggedTensor) for x in data.values()]),
        [f'All data values need to be `tf.RaggedTensor`s to be merged'])

def _assert_separable(data: NestedTensors):
    'Asserts that all nested tensors are non-ragged'
    return tf.Assert(
        all([isinstance(x, tf.Tensor) for x in data.values()]),
        [f'All data values need to be `tf.Tensor`s to be separated'])

def _slice_to_tensor(slice_obj: slice, limit: int) -> tf.Tensor:
    '''Converts slice to a tf.range, which can subsequently be used with
    tf.gather to gather subgraphs.
    '''
    start = slice_obj.start
    stop = slice_obj.stop
    step = slice_obj.step

    if stop is None:
        stop = limit
    elif stop < 0:
        stop = tf.maximum(limit + stop, 0)
    elif stop > limit:
        stop = limit

    if start is None:
        start = tf.constant(0)
    elif start < 0:
        start = tf.maximum(limit + start, 0)

    if step is None:
        step = 1
    elif step < 0:
        raise ValueError('Slice step cannot be negative')
    elif step == 0:
        raise ValueError('Slice step cannot be zero')

    start = tf.cond(start > stop, lambda: stop, lambda: start)

    return tf.range(start, stop, step)

@tf.experimental.dispatch_for_api(tf.gather)
def graph_tensor_gather(
    params: GraphTensor,
    indices,
    validate_indices=None,
    axis=None,
    batch_dims=0,
    name=None
) -> GraphTensor:
    'Gathers components (subgraphs) from graph.'

    if axis is not None and axis != 0:
        raise ValueError(
            f'axis=0 is required for `{params.__class__.__name__}`.')

    ragged = isinstance(params._data['node_feature'], tf.RaggedTensor)

    if not ragged:
        params = params.separate()

    components = tf.nest.map_structure(
        lambda x: tf.gather(
            x, indices, validate_indices, axis, batch_dims, name),
        params._data.copy())

    params = GraphTensor(components)

    if not ragged and isinstance(params._data['node_feature'], tf.RaggedTensor):
        params = params.merge()

    return params


@tf.experimental.dispatch_for_api(tf.concat)
def graph_tensor_concat(
    values: List[GraphTensor],
    axis: int = 0,
    name: str = 'concat'
) -> GraphTensor:
    'Concatenates list of graph tensors into a single graph tensor.'

    if axis is not None and axis != 0:
        raise ValueError(
            f'axis=0 is required for `{values[0].__class__.__name__}`s.')

    def get_row_lengths(x, dtype):
        if hasattr(x, 'row_lengths'):
            return tf.cast(x.row_lengths(), dtype=dtype)
        return tf.cast(x.shape[:1], dtype=dtype)

    def from_row_lengths(x, num_nodes, num_edges, dtype):
        return tf.cond(
            tf.shape(x, dtype)[0] == tf.reduce_sum(num_nodes),
            lambda: tf.RaggedTensor.from_row_lengths(x, num_nodes),
            lambda: tf.cond(
                tf.shape(x, dtype)[0] == tf.reduce_sum(num_edges),
                lambda: tf.RaggedTensor.from_row_lengths(x, num_edges),
                lambda: tf.RaggedTensor.from_row_lengths(x, tf.shape(x)[:1])
            )
        )

    structure = values[0]._data

    ragged = tf.nest.map_structure(
        lambda x: isinstance(x['node_feature'], tf.RaggedTensor), values)

    if 0 < sum(ragged) < len(ragged):
        raise ValueError(
            'The nested structure types of `values` are not the same. ' +
            'Found both nested ragged tensors and tensors of the `GraphTensor`s')
    else:
        # If first element is ragged, the rest is also ragged, and vice versa
        ragged = ragged[0]

    flat_sequence = tf.nest.map_structure(
        lambda x: tf.nest.flatten(x, expand_composites=True), values)

    dtype = values[0]['edge_dst'].dtype

    num_nodes = tf.concat([
        get_row_lengths(x['node_feature'], dtype) for x in values], axis=0)

    num_edges = tf.concat([
        get_row_lengths(x['edge_dst'], dtype) for x in values], axis=0)

    if ragged:
        # Keep only values (resulting from tf.nest.flatten)
        flat_sequence = [f[::2] for f in flat_sequence]

    flat_sequence = list(zip(*flat_sequence))
    flat_sequence = [tf.concat(x, axis=0) for x in flat_sequence]
    values = tf.nest.pack_sequence_as(structure, flat_sequence)
    values = tf.nest.map_structure(
        lambda x: from_row_lengths(x, num_nodes, num_edges, dtype),
        values)
    values = GraphTensor(values)

    if ragged:
        return values

    return values.merge()

@tf.experimental.dispatch_for_api(tf.matmul)
def graph_tensor_matmul(
    a: Union[GraphTensor, tf.Tensor],
    b: Union[GraphTensor, tf.Tensor],
    transpose_a=False,
    transpose_b=False,
    adjoint_a=False,
    adjoint_b=False,
    a_is_sparse=False,
    b_is_sparse=False,
    output_type=None
) -> tf.Tensor:
    '''Allows graph tensors to be matrix multiplied.

    Specifically, the `node_feature` field will be used for
    the matrix multiplication. This implementation makes it
    possible to pass graph tensors to `keras.layers.Dense`.
    '''
    if isinstance(a, GraphTensor):
        a = a.node_feature
    if isinstance(b, GraphTensor):
        b = b.node_feature
    return tf.matmul(
        a, b, transpose_a, transpose_b, adjoint_a,
        adjoint_b, a_is_sparse, b_is_sparse, output_type)

@tf.experimental.dispatch_for_unary_elementwise_apis(GraphTensor)
def graph_tensor_unary_elementwise_op_handler(
    api_func,
    x: GraphTensor
) -> GraphTensor:
    '''Allows all unary elementwise operations (such as `tf.math.abs`)
    to handle graph tensors.
    '''
    return x.update({'node_feature': api_func(x.node_feature)})

@tf.experimental.dispatch_for_binary_elementwise_apis(
    Union[GraphTensor, tf.Tensor],
    Union[GraphTensor, tf.Tensor])
def graph_tensor_binary_elementwise_op_handler(
    api_func,
    x: Union[GraphTensor, tf.Tensor],
    y: Union[GraphTensor, tf.Tensor],
) -> GraphTensor:
    '''Allows all binary elementwise operations (such as `tf.math.add`)
    to handle graph tensors.
    '''
    if isinstance(x, GraphTensor):
        x_values = x.node_feature
    else:
        x_values = x

    if isinstance(y, GraphTensor):
        y_values = y.node_feature
    else:
        y_values = y

    if isinstance(x, GraphTensor):
        return x.update({'node_feature': api_func(x_values, y_values)})
    elif isinstance(y, GraphTensor):
        return y.update({'node_feature': api_func(x_values, y_values)})

    return api_func(x_values, y_values)


class _Iterator:
    'Iterator for the graph tensors'

    __slots__ = ["_iterable", "_index", "_limit"]

    def __init__(self, iterable: GraphTensor, limit: int) -> None:
        self._iterable = iterable
        self._limit = limit
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._index == self._limit:
            raise StopIteration
        result = self._iterable[self._index]
        self._index += 1
        return result
