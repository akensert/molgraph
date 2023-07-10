import tensorflow as tf

from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import type_spec

try:
    from tensorflow.python.framework import type_spec_registry
except ImportError:
    type_spec_registry = None

import numpy as np

from typing import Optional
from typing import Mapping
from typing import List
from typing import Tuple
from typing import Union
from typing import Any
from typing import Type

from molgraph.layers import gnn_ops


type_spec_registry = (
    type_spec_registry.register if type_spec_registry is not None 
    else type_spec.register
)

_allowable_input_types = (
    tf.Tensor,
    tf.RaggedTensor,
    np.ndarray,
    list,
    tuple
)

_required_fields = [
    'edge_src', 
    'edge_dst', 
    'node_feature'
]

_non_updatable_fields = [
    'edge_src', 
    'edge_dst', 
    'graph_indicator'
]

GraphData = Mapping[
    str, 
    Union[ 
        tf.Tensor,
        tf.RaggedTensor,
        np.ndarray,
        list,
        tuple
    ] # TensorLike
]

GraphTensorData = Mapping[
    str, 
    Union[
        tf.Tensor, 
        tf.RaggedTensor
    ]
]

GraphTensorDataSpec = Mapping[
    str, 
    Union[
        tf.TensorSpec, 
        tf.RaggedTensorSpec
    ]
]


class GraphTensor(composite_tensor.CompositeTensor):

    '''A composite tensor encoding a molecular graph.

    The molecular graph (encoded as a :class:`~GraphTensor`) 
    could encode a single subgraph (single molecule) or multiple subgraphs 
    (multiple molecules). Furthermore, the :class:`~GraphTensor` 
    can either encode multiple molecules [molecular graphs] as a single 
    (disjoint) graph (nested "rectangular" tensors) or as multiple subgraphs 
    (nested ragged tensors). The former is advantageous for efficient 
    computation while the latter is advantageous for batching (via e.g., 
    the ``tf.data.Dataset`` API). It is recommended to :meth:`~merge` the 
    subgraphs into a single disjoint graph in the ``tf.data.Dataset`` pipeline
    before feeding the :class:`~GraphTensor` instances to
    a graph neural network (GNN) model.

    Note: every method that seemingly modifies the graph tensor instance
    actually does not modify it. Instead, a new graph tensor instance is
    returned by these methods. This is necessary as it allows TF to properly 
    track the graph tensor instances. These methods include: 
    :meth:`~merge`, :meth:`~separate`, :meth:`~update`, :meth:`~remove`.

    Args:
        data (dict):
            Nested data of the graph tensor. Specifically, a dictionary of 
            tensors (tf.Tensor or tf.RaggedTensor), numpy arrays, lists or tuples.
            Internally, values (of dict) will be converted to tensors. 
        spec: (dict):
            Nested specs (associated with nested data). Specifically, a dictionary 
            of tensor specs (tf.TensorSpec or tf.RaggedTensorSpec). Nested structure 
            of spec should be match nested structure of data.
        **data_kwargs (tf.Tensor, tf.RaggedTensor, np.array, list, tuple):
            Each nested data passed as keyword arguments.

    **Examples:**

    Initialize :class:`~GraphTensor` instance by passing a dict of ragged arrays,
    resulting in a graph tensor instance with nested ``tf.RaggedTensor`` types:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     data={
    ...         'edge_src': [[1, 0], [1, 2, 0, 2, 1, 0]],
    ...         'edge_dst': [[0, 1], [0, 0, 1, 1, 2, 2]],
    ...         'node_feature': [
    ...             [[1.0, 0.0], [1.0, 0.0]],
    ...             [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    ...         ],
    ...     }
    ... )
    >>> graph_tensor.shape
    TensorShape([2, None, 2])

    Initialize :class:`~GraphTensor` instance by passing a dict of "rectangular" 
    arrays, resulting in a graph tensor instance with nested ``tf.Tensor`` types:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     data={
    ...         'edge_src': [1, 0, 3, 4, 2, 4, 3, 2],
    ...         'edge_dst': [0, 1, 2, 2, 3, 3, 4, 4],
    ...         'node_feature': [
    ...             [1.0, 0.0],
    ...             [1.0, 0.0],
    ...             [1.0, 0.0],
    ...             [1.0, 0.0],
    ...             [0.0, 1.0]
    ...         ],
    ...         'graph_indicator': [0, 0, 1, 1, 1],
    ...     }
    ... )
    >>> graph_tensor.shape
    TensorShape([5, 2])

    Initialize :class:`~GraphTensor` instance by passing data as keyword 
    arguments:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     edge_src=[[1, 0], [1, 2, 0, 2, 1, 0]],
    ...     edge_dst=[[0, 1], [0, 0, 1, 1, 2, 2]],
    ...     node_feature=[
    ...         [[1.0, 0.0], [1.0, 0.0]],
    ...         [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    ...     ]
    ... )
    >>> graph_tensor.shape
    TensorShape([2, None, 2])

    Merge, separate and merge again the subgraphs of :class:`~GraphTensor`:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     edge_src=[[1, 0], [1, 2, 0, 2, 1, 0]],
    ...     edge_dst=[[0, 1], [0, 0, 1, 1, 2, 2]],
    ...     node_feature=[
    ...         [[1.0, 0.0], [1.0, 0.0]],
    ...         [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    ...     ]
    ... )
    >>> graph_tensor = graph_tensor.merge()    # nesterd tensors
    >>> graph_tensor = graph_tensor.separate() # nested ragged tensors
    >>> graph_tensor = graph_tensor.merge()    # nested tensors
    >>> graph_tensor.shape
    TensorShape([5, 2])

    Add, update and remove data from :class:`~GraphTensor``:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     data={
    ...         'edge_src': [1, 0, 3, 4, 2, 4, 3, 2],
    ...         'edge_dst': [0, 1, 2, 2, 3, 3, 4, 4],
    ...         'node_feature': [
    ...             [1.0, 0.0],
    ...             [1.0, 0.0],
    ...             [1.0, 0.0],
    ...             [1.0, 0.0],
    ...             [0.0, 1.0]
    ...         ],
    ...         'graph_indicator': [0, 0, 1, 1, 1],
    ...     }
    ... )
    >>> random_feature_1 = tf.random.uniform(
    ...     graph_tensor['node_feature'].shape)
    >>> random_feature_2 = tf.random.uniform(
    ...    graph_tensor['node_feature'].shape)
    >>> # Add new data
    >>> graph_tensor = graph_tensor.update({
    ...     'node_random_feature': random_feature_1})
    >>> # Update exisiting data
    >>> graph_tensor = graph_tensor.update({
    ...     'node_feature': random_feature_2})
    >>> # Remove data
    >>> graph_tensor = graph_tensor.remove(['node_random_feature'])
    >>> graph_tensor
    GraphTensor(
      edge_src=<tf.Tensor: shape=(8,), dtype=int32>,
      edge_dst=<tf.Tensor: shape=(8,), dtype=int32>,
      node_feature=<tf.Tensor: shape=(5, 2), dtype=float32>,
      graph_indicator=<tf.Tensor: shape=(5,), dtype=int32>)

    Use spec of ``GraphTensor`` in ``keras.Sequential`` model:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     data={
    ...         'edge_src': [1, 0, 3, 4, 2, 4, 3, 2],
    ...         'edge_dst': [0, 1, 2, 2, 3, 3, 4, 4],
    ...         'node_feature': [
    ...             [1.0, 0.0],
    ...             [1.0, 0.0],
    ...             [1.0, 0.0],
    ...             [1.0, 0.0],
    ...             [0.0, 1.0]
    ...         ],
    ...         'graph_indicator': [0, 0, 1, 1, 1],
    ...     }
    ... )
    >>> # Build a model with .spec (not recommended)
    >>> gnn_model = tf.keras.Sequential([
    ...     # tf.keras.Input(type_spec=graph_tensor.spec),
    ...     # Recommended to use unspecific_spec here instead:
    ...     tf.keras.Input(type_spec=graph_tensor.unspecific_spec),
    ...     molgraph.layers.GCNConv(16, activation='relu'),
    ...     molgraph.layers.GCNConv(16, activation='relu')
    ... ])
    >>> gnn_model.output_shape
    (None, 16)
    '''

    __slots__ = ('_data', '_spec')

    def __init__(
        self,
        data: Optional[GraphData] = None,
        spec: Optional[GraphTensorDataSpec] = None,
        **data_kwargs 
    ) -> None:
        
        super().__init__()

        if data is None:
            data = {}

        data.update(data_kwargs)

        data = _convert_to_tensors(
            data, check_keys=True, check_values=True)

        if spec is None:
            spec = tf.nest.map_structure(tf.type_spec_from_value, data)
        else:
            tf.nest.assert_same_structure(data, spec)

        data, spec = _maybe_add_graph_indicator(data, spec)

        self._spec = GraphTensorSpec(spec)
        self._data = data
    
    def update(
        self, 
        new_data: Optional[GraphData] = None, 
        **new_data_kwargs
    ) -> 'GraphTensor':
        '''Updates existing data or adds new data to GraphTensor instance.

        Constraints are put on the update method: 

            * New data needs to match the size of existing data (i.e., it has 
              to have the same number of nodes or edges as the existing data).
            * Furthermore, names of new data need to prepend either 'node' (if 
              associated with nodes) or 'edge' (if associated with edges) to 
              allow the graph tensor to keep track of what data belong to which. 

        Edge case:

            *   Although very rare, avoid updating a ragged graph tensor 
                instance with non-ragged (flat) values from another completely 
                different graph. This other graph [graph tensor instance] may 
                have the same number of nodes (or edges) but different row 
                lengths (i.e. different sized subgraphs (indicated by the graph 
                indicator)). The problem is that the ragged graph tensor 
                instance will happily accept the new data (e.g. node features
                of this other graph), but incorrectly partition them (via 
                ``with_flat_values()``). 

        Args:
            new_data (dict):
                Nested data. Specifically, a dictionary of tensors (either 
                ``tf.Tensor`` or ``tf.RaggedTensor``), `np.array`s, ``list``s 
                or ``tuple``s.

        Returns:
            An updated ``GraphTensor`` instance.
        '''
        def convert_value(
            new_value: Union[tf.Tensor, tf.RaggedTensor],
            old_value: Union[tf.Tensor, tf.RaggedTensor],
        ) -> Union[tf.Tensor, tf.RaggedTensor]:
            
            if (
                isinstance(new_value, tf.RaggedTensor) and
                isinstance(old_value, tf.Tensor)
            ):
                new_value = new_value.flat_values
                _check_shape(old_value, new_value)
            elif (
                isinstance(new_value, tf.Tensor) and  
                isinstance(old_value, tf.RaggedTensor)
            ):
                new_value = old_value.with_flat_values(new_value)
                # No need to assert shape as the method will throw an error 
                # if shapes are mismatching.
            else:
                new_value = new_value
                _check_shape(old_value, new_value)

            return new_value

        if new_data is None:
            new_data = {}

        new_data.update(new_data_kwargs)

        new_data = _convert_to_tensors(new_data, check_values=True)

        data = self._data.copy()

        fields = list(new_data.keys())

        for field in fields:

            # TODO: should we allow edge_src, edge_dst and graph_indicator to be updatable?
            if field in _non_updatable_fields:
                raise ValueError(f'{field} cannot be updated.')

            new_value = new_data.pop(field)

            if field in data:
                data[field] = convert_value(new_value, data[field])
            else:
                if not field.startswith('node') and not field.startswith('edge'):
                    raise ValueError(
                        'Please prepend "node" or "edge" to the new data added, '
                        'depending on if they are associated with the nodes or edges '
                        'of the graph respectively.'
                    )
                elif field.startswith('edge'):
                    data[field] = convert_value(new_value, data['edge_src'])
                else:
                    data[field] = convert_value(new_value, data['node_feature'])

        return self.__class__(data)
    
    def remove(
        self,
        fields: Union[str, List[str]]
    ) -> 'GraphTensor':
        '''Removes data from the graph tensor instance.

        Args:
            fields (str, list[str]):
                Data to be removed from the graph tensor. Currently, 
                'edge_dst', 'edge_src' or 'graph_indicator' cannot be removed.

        Returns:
            GraphTensor: An updated graph tensor instance.
        '''
        data = self._data.copy()

        if isinstance(fields, str):
            fields = [fields]

        for field in fields:
            
            if field in _non_updatable_fields:
                raise ValueError(f'{field} cannot be removed.')
        
            data.pop(field)

        return self.__class__(data)

    def merge(self) -> 'GraphTensor':
        '''Merges subgraphs into a single disjoint graph.

        Returns:
            GraphTensor: A graph tensor instance with nested "rectangular" 
            tensors.
        '''
        _check_mergeable(self._data)

        data = self._data.copy()

        increment = data['node_feature'].row_starts()
        indices = data['edge_src'].value_rowids()
        graph_indicator = data['node_feature'].value_rowids()
        increment = tf.cast(increment, dtype=data['edge_src'].dtype)
        data = tf.nest.map_structure(lambda x: x.flat_values, data)
        data['edge_src'] += tf.gather(increment, indices)
        data['edge_dst'] += tf.gather(increment, indices)
        data['graph_indicator'] = graph_indicator

        return self.__class__(data)

    def separate(self) -> 'GraphTensor':
        '''Separates the (single disjoint) graph into its subgraphs.

        Returns:
            GraphTensor: A graph tensor instance with nested ragged tensors.
        '''

        def to_ragged_tensor(
            tensor: Union[tf.Tensor, tf.RaggedTensor],
            graph_indicator: tf.Tensor,
            num_subgraphs: tf.Tensor,
        ) -> tf.RaggedTensor:
            if isinstance(tensor, tf.RaggedTensor):
                return tensor
            return tf.RaggedTensor.from_value_rowids(
                tensor, graph_indicator, num_subgraphs)
        

        _check_separable(self._data)

        data = self._data.copy()

        if 'graph_indicator' not in data:
            return self.__class__(
                tf.nest.map_structure(
                    lambda x: tf.RaggedTensor.from_row_starts(x, [0]), data
                )
            )

        data = _remove_intersubgraph_edges(data)

        graph_indicator = data.pop('graph_indicator')
        edge_src = data.pop('edge_src')
        edge_dst = data.pop('edge_dst')
        graph_indicator_edges = tf.gather(graph_indicator, edge_src)
        num_subgraphs = self.num_subgraphs

        edge_data, node_data = {}, {}
        for key, value in data.items():
            if key.startswith('edge'):
                edge_data[key] = value
            else:
                node_data[key] = value

        edge_data = tf.nest.map_structure(
            lambda x: to_ragged_tensor(
                x, graph_indicator_edges, num_subgraphs), edge_data)
        
        node_data = tf.nest.map_structure(
            lambda x: to_ragged_tensor(
                x, graph_indicator, num_subgraphs), node_data)   

        decrement = tf.gather(
            node_data['node_feature'].row_starts(), graph_indicator_edges)
        decrement = tf.cast(decrement, dtype=edge_src.dtype)
        edge_data['edge_src'] = tf.RaggedTensor.from_value_rowids(
            edge_src - decrement, graph_indicator_edges, num_subgraphs)
        edge_data['edge_dst'] = tf.RaggedTensor.from_value_rowids(
            edge_dst - decrement, graph_indicator_edges, num_subgraphs)
        
        return self.__class__({**node_data, **edge_data})
    
    def propagate(
        self, 
        mode: Optional[str] = 'sum',
        normalize: bool = False,
        reduction: Optional[str] = None,
        residual: Optional[tf.Tensor] = None,
        **kwargs,
    ):
        '''Propagates node features.

        This is a helper method for passing information between nodes;
        specifically, it aggregates information (features) from source
        nodes to destination nodes. Roughly, this method uses three
        :mod:`~molgraph.layers.gnn_ops` in sequence, the first and third 
        being optional:

        (1) normalizes edge weights via 
            :func:`~molgraph.layers.gnn_ops.softmax_edge_weights`;
        (2) propagates node features via 
            :func:`~molgraph.layers.gnn_ops.propagate_node_features`;
        (3) reduces aggregated node features via 
            :func:`~molgraph.layers.gnn_ops.reduce_features`. 
        
        Args:
            mode (str):
                The type of aggregation to be performed, either of 'sum', 'mean',
                'min' or 'max'. If None, 'sum' will be used. Default to 'sum'.
            normalize (bool):
                Whether the edge weights (if available) should be normalized 
                (via softmax) before aggregation. Edge weights are usually the 
                attention scores applied to each incoming (source) node feature.
            reduction (None, str):
                The type of reduction ("merging") to be performed if the node 
                features span another dimension (e.g. when using multiple 
                attention heads in :class:`~molgraph.layers.GATConv` or 
                :class:`~molgraph.layers.GTConv`). Either of 'concat', 'mean', 
                'sum' or None. Default to None. 
            residual (None, tf.Tensor):
                Residual node features to be added to the output of the 
                aggregated node features. Default to None.
            
            **kwargs: Valid (optional) keyword arguments are:
            
                *   `activation`: The activation to be performed on the 
                    aggregated node features. Default to None.
                *   `exponentiate`: Whether to exponentiate edge weights before
                    softmax (defualt to True).
                *   `clip_values`: The clipping range that should be applied to
                    the (potentially exponentiated) edge weights. For stability.
                    (default to True).
                *   `output_units`: the output dimension (innermost dimension) 
                    after reshaping. Only relevant if ``reduction='concat'``. 
                    Default to None.
                *   `reduce_axis`: Axis to be reduced ("merged"). Ignored
                    if None. Default to 1, which is the axis of the heads
                    of :class:`~molgraph.layers.GATConv`, 
                    :class:`~molgraph.layers.GTConv` etc.
        '''
        exponentiate = kwargs.get('exponentiate', True)
        clip_values = kwargs.get('clip_values', (-5., 5.))
        activation = kwargs.get('activation', None)
        output_units = kwargs.get('output_units', None)
        reduce_axis = kwargs.get('reduce_axis', 1)

        if self.is_ragged():
            data = self.merge()._data.copy()
        else:
            data = self._data.copy()

        if normalize and 'edge_weight' in data:

            data['edge_weight'] = gnn_ops.softmax_edge_weights(
                edge_weight=data['edge_weight'], 
                edge_dst=data['edge_dst'], 
                exponentiate=exponentiate,
                clip_values=clip_values)
            
        data['node_feature'] = gnn_ops.propagate_node_features(
            node_feature=data['node_feature'],
            edge_src=data['edge_src'],
            edge_dst=data['edge_dst'],
            edge_weight=data.get('edge_weight', None),
            mode=mode)
        
        if residual is not None:
            if isinstance(residual, tf.RaggedTensor):
                residual = residual.flat_values
            data['node_feature'] += residual

        if activation is not None:
            data['node_feature'] = activation(data['node_feature'])

        if reduction:
            data['node_feature'] = gnn_ops.reduce_features(
                feature=data['node_feature'], 
                mode=reduction,
                output_units=output_units,
                axis=reduce_axis)
        
        graph_tensor = self.__class__(data)

        if not self.is_ragged():
            return graph_tensor 
        return graph_tensor.separate()

    def is_ragged(self):
        '''Checks whether nested data are ragged.
        
        Returns:
            bool: A boolean indicating whether nested data are ragged.
        '''
        return isinstance(self._data['node_feature'], tf.RaggedTensor)
    
    @property
    def _type_spec(self) -> 'GraphTensorSpec':
        'CompositeTensor API.'
        return self._spec

    @property
    def spec(self) -> 'GraphTensorSpec':
        '''Spec of the graph tensor instance.

        It is recommended to use :meth:`~unspecific_spec` instead of 
        :meth:`~spec`, especially when using the ``tf.saved_model`` API.

        Returns:
            GraphTensorSpec: the corresponding spec of the graph tensor 
            instance.
        '''
        return self._type_spec

    @property
    def unspecific_spec(self):
        '''Unspecific spec of graph tensor instance.

        Specifically, batch dimension for all nested data specs are set to None.

        It is recommended to use :meth:`~unspecific_spec` instead of 
        :meth:`~spec`, especially when using the ``tf.saved_model`` API.

        Returns:
            GraphTensorSpec: the corresponding (unspecific) spec of the graph 
            tensor instance.
        '''
        return _unspecify_batch_shape(self._type_spec)

    @property
    def shape(self) -> tf.TensorShape:
        'Partial shape of the graph tensor instance (based on ``node_feature``).'
        return self._spec.shape

    @property
    def dtype(self) -> tf.DType:
        'Partial dtype of the graph tensor instance (based on ``node_feature``).'
        return self._spec.dtype

    @property
    def rank(self) -> int:
        'Partial rank of the graph tensor instance (based on ``node_feature``).'
        return self._spec.rank

    @property
    def num_subgraphs(self):
        if 'graph_indicator' in self._data:
            return tf.maximum(
                tf.math.reduce_max(self._data['graph_indicator']) + 1, 0) 
        elif isinstance(self._data['node_feature'], tf.RaggedTensor):
            return self._data['node_feature'].nrows()
        else:
            # TODO: return None?
            return tf.constant(1, dtype=self._data['edge_src'].dtype)
    
    @property
    def node_feature(self):
        'Obtain `node_feature` from graph tensor instance.'
        return self._data.get('node_feature', None) 
    
    @property
    def edge_src(self):
        'Obtain `edge_src` from graph tensor instance.'
        return self._data.get('edge_src', None) 
    
    @property
    def edge_dst(self):
        'Obtain `edge_dst` from graph tensor instance.'
        return self._data.get('edge_dst', None) 
    
    @property
    def edge_feature(self):
        'Obtain `edge_feature` from graph tensor instance.'
        return self._data.get('edge_feature', None) 
    
    @property 
    def edge_weight(self):
        'Obtain `edge_weight` from graph tensor instance.'
        return self._data.get('edge_weight', None) 
    
    @property
    def graph_indicator(self):
        'Obtain `graph_indicator` from graph tensor instance.'
        return self._data.get('graph_indicator', None) 
    
    def __getattr__(self, name: str) -> Union[tf.Tensor, tf.RaggedTensor, Any]:
        '''Access nested data as attributes.

        Only called when attribute lookup has not found attribute `name` in
        the usual places. Hence convenient when new data has been added to
        the graph tensor and we want to access this data as an attribute.

        Args:
            name (str):
                The data to be extracted; e.g., ``graph_tensor.node_feature``

        Returns:
            If ``name`` in nested data, data is returned. Otherwise,
            attribute ``name`` of the graph tensor instance is returned.
        
        Raises:
            AttributeError: if `name` does not exist in data.
        '''
        if name in object.__getattribute__(self, '_data'):
            return self._data[name]
        
        _raise_attribute_error(self, name)

    def __getitem__(
        self,
        index: Union[str, int, List[int]]
    ) -> Union[tf.RaggedTensor, tf.Tensor, 'GraphTensor']:
        '''Access nested data or subgraphs via indexing.

        Args:
            index (str, int, list[int]):
                If str, extracts specific data from graph tensor instance; 
                if int or list[int], extracts subgraph(s) from the graph
                tensor instance.

        Returns:
            A :class:`~GraphTensor` instance with specified subgraphs, or a
            ``tf.Tensor`` or ``tf.RaggedTensor`` holding the specified data.
        
        Raises:
            KeyError: if `index` (str) does not exist in data spec.
            tf.errors.InvalidArgumentError: if `index` (int, list[int]) is out 
            of range.
        '''
        if isinstance(index, str):
            return self._data[index]
        if isinstance(index, slice):
            index = _slice_to_tensor(index, self.num_subgraphs)
        return tf.gather(self, index)

    def __iter__(self):
        if not tf.executing_eagerly():
            raise ValueError(
                'Can only iterate over `GraphTensor` in eager mode.')
        return _Iterator(self, limit=self.num_subgraphs)

    def __repr__(self) -> str:
        fields = []
        for key, value in self._spec._data_spec.items():
            if isinstance(self._data[key], tf.RaggedTensor):
                # Include value.ragged_rank and value.row_splits_dtype.name?
                fields.append(
                    '{}=<tf.RaggedTensor: '.format(key) +
                    'shape={}, '.format(value.shape) +
                    'dtype={}>'.format(value.dtype.name)
                    # TODO: include e.g. ragged_rank?
                )
            elif isinstance(self._data[key], tf.Tensor):
                fields.append(
                    '{}=<tf.Tensor: '.format(key) +
                    'shape={}, '.format(value.shape) +
                    'dtype={}>'.format(value.dtype.name)
                )
            else:
                # Should not happen, but just in case.
                fields.append('{}=<unknown>'.format(key))

        return f'GraphTensor(\n  ' + ',\n  '.join(fields) + ')'


@type_spec_registry('molgraph.tensors.graph_tensor.GraphTensorSpec')
class GraphTensorSpec(type_spec.BatchableTypeSpec):

    '''Spec of :class:`~GraphTensor`.
    '''
    __slots__ = ('_data_spec', '_shape', '_dtype')

    def __init__(
        self,
        data_spec: GraphTensorDataSpec,
        shape: Optional[tf.TensorShape] = None,
        dtype: Optional[tf.DType] = None,
    ) -> None:
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
        # ExtensionType API
        return GraphTensor
    
    @property
    def node_feature(self):
        'Obtain `node_feature` spec from graph tensor spec.'
        return self._data_spec.get('node_feature', None) 
    
    @property
    def edge_src(self):
        'Obtain `edge_src` spec from graph tensor spec.'
        return self._data_spec.get('edge_src', None) 
    
    @property
    def edge_dst(self):
        'Obtain `edge_dst` spec from graph tensor spec.'
        return self._data_spec.get('edge_dst', None) 
    
    @property
    def edge_feature(self):
        'Obtain `edge_feature` spec from graph tensor spec.'
        return self._data_spec.get('edge_feature', None) 
    
    @property 
    def edge_weight(self):
        'Obtain `edge_weight` spec from graph tensor spec.'
        return self._data_spec.get('edge_weight', None) 
    
    @property
    def graph_indicator(self):
        'Obtain `graph_indicator` spec from graph tensor spec.'
        return self._data_spec.get('graph_indicator', None) 
    
    def __getattr__(self, name: str) -> Union[tf.Tensor, tf.RaggedTensor, Any]:
        '''Access nested data spec as attributes.

        Only called when attribute lookup has not found attribute `name` in
        the usual places. Hence convenient when new data has been added to
        the corresponding graph tensor and we want to access this data as 
        an attribute.

        Args:
            name (str):
                The data spec to be extracted; e.g., 
                ``graph_tensor_spec.node_feature``

        Returns:
            If ``name`` in nested data spec, data spec is returned. Otherwise,
            attribute ``name`` of the graph tensor instance is returned.
        
        Raises:
            AttributeError: if `name` does not exist in data spec.
        '''    
        if name in object.__getattribute__(self, '_data'):
            return self._data_spec[name]
        
        _raise_attribute_error(self, name)

    def __getitem__(
        self,
        name: str
    ) -> Union[tf.RaggedTensorSpec, tf.TensorSpec]:
        '''Access nested data spec via indexing.

        Args:
            index (str):
                The data spec to be extracted; e.g., 
                ``graph_tensor_spec['node_feature']``

        Returns:
            ``tf.TensorSpec`` or ``tf.RaggedTensorSpec`` holding the 
            specified data.

        Raises:
            KeyError: if `name` does not exist in data spec.
        '''
        return self._data_spec[name]
    
    @property
    def shape(self) -> tf.TensorShape:
        'Partial shape of spec (based on ``node_feature``).'
        return self._shape

    @property
    def dtype(self) -> tf.DType:
        'Partial dtype of spec (based on ``node_feature``).'
        return self._dtype

    @property
    def rank(self) -> int:
        'Partial rank of spec (based on ``node_feature``).'
        return self._shape.rank

    def with_shape(self, shape: tf.TensorShape) -> 'GraphTensorSpec':
        # Keras API
        return self.__class__(self._data_spec, shape, self._dtype)

    @classmethod
    def from_value(cls, value: GraphTensor) -> 'GraphTensorSpec':
        # ExtensionType API
        return value._type_spec

    def _serialize(
        self
    ) -> Tuple[GraphTensorDataSpec, tf.TensorShape, tf.DType]:
        # ExtensionType API
        return (self._data_spec, self._shape, self._dtype)

    @classmethod
    def _deserialize(
        cls,
        serialization: Tuple[
            GraphTensorDataSpec,
            Union[tf.TensorShape, None],
            Union[tf.DType, None]
        ]
    ) -> 'GraphTensorSpec':
        # ExtensionType API
        data_spec, shape, dtype = serialization
        return cls(data_spec, shape, dtype)

    @property
    def _component_specs(self) -> GraphTensorDataSpec:
        # ExtensionType API
        return self._data_spec

    def _to_components(self, value: GraphTensor) -> GraphTensorData:
        # ExtensionType API
        return value._data.copy()

    def _from_components(self, components: GraphTensorData) -> GraphTensor:
        # ExtensionType API
        return self.value_type(components, self._data_spec)

    def _batch(self, batch_size: Union[int, None]) -> 'GraphTensorSpec':
        # BatchableExtensionType API
        batched_data_spec = tf.nest.map_structure(
            lambda spec: spec._batch(batch_size), self._data_spec)
        shape = tf.TensorShape([batch_size]).concatenate(self._shape)
        return self.__class__(batched_data_spec, shape, self._dtype)

    def _unbatch(self) -> 'GraphTensorSpec':
        # BatchableExtensionType API
        unbatched_data_spec = tf.nest.map_structure(
            lambda spec: spec._unbatch(), self._data_spec)
        shape = self._shape[1:]
        return self.__class__(unbatched_data_spec, shape, self._dtype)

    def _to_legacy_output_types(self):
        # Legacy method of ExtensionType API
        return self._dtype

    def _to_legacy_output_shapes(self):
        # Legacy method of ExtensionType API
        return self._shape

    def _to_legacy_output_classes(self):
        # Legacy method of ExtensionType API
        return self
    

def _raise_attribute_error(instance, name):
    class_name = instance.__class__.__name__
    raise AttributeError(f'{class_name!r} object has no attribute {name!r}')

def _assert(test: bool, message: str) -> None:
    'Helper function to make assert statements.'
    assert_op = tf.Assert(tf.reduce_all(test), [message])
    if hasattr(assert_op, 'mark_used'):
        assert_op.mark_used()

def _check_shape(
    a: Union[tf.Tensor, tf.RaggedTensor],
    b: Union[tf.Tensor, tf.RaggedTensor],
) -> None:
    'Assert that a and b have the same number of nodes (or edges)'

    if tf.executing_eagerly():
        _assert(type(a) == type(b), ['a and b need to be the same type'])

    if isinstance(a, tf.Tensor):
        _assert(
            test=(tf.shape(a)[0] == tf.shape(b)[0]), 
            message=(
                'The shape of input `a` does not match the shape of input `b`'
            )
        )
    else:
        _assert(
            test=(tf.shape(a)[:2] == tf.shape(b)[:2]), 
            message=(
                'The shape of input `a` does not match the shape of input `b`'
            )
        )

def _check_tensor_types(data: GraphTensorData) -> None:
    'Assert that all graph data are of the same tensor type.'
    tests = [isinstance(x, tf.Tensor) for x in data.values()]
    same_types = all(tests) or not any(tests)
    _assert(
        same_types, (
            f'Nested tensors are not the same type. ' +
             'Found both `tf.Tensor`s and `tf.RaggedTensor`s'
        )
    )

def _check_tensor_ranks(data: GraphTensorData) -> None:
    'Assert that all graph data have the expected rank.'
    for key, value in data.items():
        max_rank = 1 if isinstance(value, tf.Tensor) else 2
        if key in ['edge_dst', 'edge_src', 'graph_indicator']:
            _assert(tf.rank(value) <= max_rank, '')
        elif key in ['node_feature', 'edge_feature']:
            _assert(tf.rank(value) <= max_rank+2, '')
        else:
            _assert(tf.rank(value) <= max_rank+2, '')


def _check_data_keys(data: GraphData) -> None:
    'Assert that required graph data exist.'
    for req_field in _required_fields:
        _assert(
            req_field in data, 
            f'`data` requires `{req_field}` field'
        )

def _check_data_values(data: GraphData) -> None:
    'Assert that all inputted graph data have the expected type'
    for key, value in data.items():
        _assert(
            isinstance(value, _allowable_input_types), (
                f'Field `{key}` is needs to be a `tf.Tensor`, ' +
                '`tf.RaggedTensor`, `np.ndarray`, `list` or `tuple`'
            )
        )

def _check_mergeable(data: GraphTensorData) -> None:
    'Assert that all nested tensors are ragged.'
    all_ragged = all([
        isinstance(x, tf.RaggedTensor) for x in data.values()
    ])
    _assert(all_ragged, (
            'All data values need to be `tf.RaggedTensor`s to be merged.'
        )
    )

def _check_separable(data: GraphTensorData) -> None:
    'Assert that all nested tensors are non-ragged'
    all_non_ragged = all([
        isinstance(x, tf.Tensor) for x in data.values()
    ])
    _assert(all_non_ragged, (
            'All data values need to be `tf.Tensor`s to be separated.'
        )
    )

def _convert_to_tensors(
    data: GraphData,
    check_values: bool = False,
    check_keys: bool = False,
) -> GraphTensorData:
    'Converts graph data (possibly ``np.array``s, ``list``s or ``tuple``s) to tensors.'

    if check_keys:
        _check_data_keys(data)

    if check_values:
        _check_data_values(data)

    def to_tensor(x):
        if not tf.is_tensor(x):
            try:
                return tf.convert_to_tensor(x)
            except:
                # TODO: slow; implement something like ``fast_convert_to_ragged()```
                return tf.ragged.constant(x, ragged_rank=1)
        else:
            return x
        
    data = {k: to_tensor(v) for (k, v) in data.items()}

    _check_tensor_types(data)
    _check_tensor_ranks(data)

    return data

def _maybe_add_graph_indicator(
    data: GraphTensorData,
    spec: GraphTensorDataSpec,
) -> GraphTensorData:
    'Adds a `graph_indicator` if necessary.'
    if (
        'graph_indicator' not in data 
        and isinstance(data['node_feature'], tf.Tensor) 
        and not isinstance(spec['node_feature'], tf.RaggedTensorSpec)
    ):
        data['graph_indicator'] = tf.zeros(
            tf.shape(data['node_feature'])[0],
            dtype=data['edge_src'].dtype)
        spec['graph_indicator'] = tf.type_spec_from_value(
            data['graph_indicator'])
        if isinstance(spec['edge_src'], tf.RaggedTensorSpec):
            x = spec['graph_indicator']
            spec['graph_indicator'] = tf.RaggedTensorSpec(
                x.shape, x.dtype, spec['edge_src'].ragged_rank, spec['edge_src'].row_splits_dtype
            )
    return data, spec

def _remove_intersubgraph_edges(data: GraphTensorData) -> GraphTensorData:
    '''Removes edges that connects two different subgraphs.

    Applied only when graph_tensor.separate() is called, 
    wherein we are "forced" to remove them.
    '''
    subgraphs_src = tf.gather(data['graph_indicator'], data['edge_src'])
    subgraphs_dst = tf.gather(data['graph_indicator'], data['edge_dst'])
    # Fine the intersubgraph edges:
    mask = tf.where((subgraphs_dst - subgraphs_src) == 0, True, False)
    # Remove these edges (including data that is associated with edges):
    for key in data.keys():
        if key.startswith('edge'):
            data[key] = tf.boolean_mask(data[key], mask)
    return data

def _unspecify_batch_shape(graph_tensor_spec: GraphTensorSpec):

    def fn(spec):
        if isinstance(spec, tf.RaggedTensorSpec):
            return tf.RaggedTensorSpec(
                shape=tf.TensorShape([None]).concatenate(spec.shape[1:]),
                dtype=spec.dtype,
                row_splits_dtype=spec.row_splits_dtype,
                ragged_rank=spec.ragged_rank,
                flat_values_spec=spec.flat_values_spec)
        return tf.TensorSpec(
            shape=tf.TensorShape([None]).concatenate(spec.shape[1:]),
            dtype=spec.dtype)
    
    return graph_tensor_spec.__class__(
        tf.nest.map_structure(fn, graph_tensor_spec._data_spec))
    
def _slice_to_tensor(slice_obj: slice, limit: int) -> tf.Tensor:
    '''Converts slice to a tf.range, which can subsequently be used with
    tf.gather to gather subgraphs.

    Note: GraphTensor is currently irreversible (e.g., x[::-1] or x[::-2]
    will not work).
    '''
    start = slice_obj.start
    stop = slice_obj.stop
    step = slice_obj.step

    _assert(
        step is None or not (step < 0 or step == 0),
        'Slice step cannot be negative or zero.'
    )

    limit = tf.convert_to_tensor(limit)

    if stop is None:
        stop = limit
    else:
        stop = tf.convert_to_tensor(stop)
        stop = tf.cond(
            stop < 0,
            lambda: tf.maximum(limit + stop, 0),
            lambda: tf.cond(
                stop > limit,
                lambda: limit,
                lambda: stop
            )
        )

    if start is None:
        start = tf.constant(0)
    else:
        start = tf.convert_to_tensor(start)
        start = tf.cond(
            start < 0,
            lambda: tf.maximum(limit + start, 0),
            lambda: start
        )

    if step is None:
        step = tf.constant(1)
    else:
        step = tf.convert_to_tensor(step)

    start = tf.cond(start > stop, lambda: stop, lambda: start)

    return tf.range(start, stop, step)


# Make GraphTensor iterable:

class _Iterator:

    __slots__ = ['_iterable', '_index', '_limit']

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


# Override default tensorflow implemenations for GraphTensor:

@tf.experimental.dispatch_for_api(tf.shape)
def graph_tensor_tf_shape(
    input: GraphTensor,
    out_type=tf.dtypes.int32,
    name=None
):
    node_feature = input.node_feature

    # TODO: Hopefully, these two lines can be removed in the future.
    # As of now, this is necessary to allow ragged GraphTensor 
    # to be used with keras.Model.predict. If DynamicRaggedShape 
    # is returned, this will throw an error: tf.shape(tensor)[1:]
    if isinstance(node_feature, tf.RaggedTensor):
        node_feature = node_feature.flat_values

    return tf.shape(
        input=node_feature, out_type=out_type, name=name)
        
@tf.experimental.dispatch_for_api(tf.gather)
def graph_tensor_gather(
    params: GraphTensor,
    indices,
    validate_indices=None,
    axis=None,
    batch_dims=0,
    name=None
) -> GraphTensor:
    'Gathers subgraphs from graph.'

    if axis is not None and axis != 0:
        raise ValueError(
            f'axis=0 is required for `{params.__class__.__name__}`.')

    ragged = isinstance(params._data['node_feature'], tf.RaggedTensor)

    if not ragged:
        params = params.separate()

    data = tf.nest.map_structure(
        lambda x: tf.gather(
            x, indices, validate_indices, axis, batch_dims, name),
        params._data.copy())

    params = GraphTensor(data)

    if not ragged and isinstance(params._data['node_feature'], tf.RaggedTensor):
        params = params.merge()

    return params

@tf.experimental.dispatch_for_api(tf.concat)
def graph_tensor_concat(
    values: List[GraphTensor],
    axis: int = 0,
    name: str = 'concat'
) -> GraphTensor:
    '''Concatenates list of graph tensors into a single graph tensor.
    
    This is important for tf.keras.Model.predict, as it concatenates
    the batches (of possibly `GraphTensor`s).
    '''

    if axis is not None and axis != 0:
        raise ValueError(
            f'axis=0 is required for `{values[0].__class__.__name__}`s.')

    def fast_ragged_stack(inputs, dtype):
        row_lengths = [tf.shape(x)[0] for x in inputs]
        inputs_concat = tf.concat(inputs, axis=0)
        return tf.RaggedTensor.from_row_lengths(
            inputs_concat, tf.cast(row_lengths, dtype))

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
    
    dtype = values[0]['edge_src'].dtype

    if ragged:
        # Keep only values (resulting from tf.nest.flatten)
        row_splits = [f[1::2] for f in flat_sequence]
        flat_sequence = [f[0::2] for f in flat_sequence]
        flat_sequence = tf.nest.map_structure(
            lambda v, r: tf.RaggedTensor.from_row_splits(v, r),
            flat_sequence, row_splits)
        
    flat_sequence = list(zip(*flat_sequence))

    if ragged:
        flat_sequence_stacked = [
            tf.concat(x, axis=0) for x in flat_sequence
        ]
    else:
        flat_sequence_stacked = [
            fast_ragged_stack(x, dtype) for x in flat_sequence
        ]

    values = tf.nest.pack_sequence_as(structure, flat_sequence_stacked)

    values = GraphTensor(values)

    if ragged:
        return values

    return values.merge()

@tf.experimental.dispatch_for_api(tf.stack)
def graph_tensor_stack(
    values: List[GraphTensor],
    axis: int = 0,
    name: str = 'stack'
) -> GraphTensor:
    '''Stacks list of graph tensors into a ragged GraphTensor.

    Note: tf.stack(list_of_graph_tensors) only works (and make sense) 
    if the graph data is non-ragged (namely, tf.Tensor types).
    '''

    if axis is not None and axis != 0:
        raise ValueError(
            f'axis=0 is required for `{values[0].__class__.__name__}`s.')

    def fast_ragged_stack(inputs, dtype):
        row_lengths = [tf.shape(x)[0] for x in inputs]
        inputs_concat = tf.concat(inputs, axis=0)
        return tf.RaggedTensor.from_row_lengths(
            inputs_concat, tf.cast(row_lengths, dtype))

    structure = values[0]._data

    ragged = tf.nest.map_structure(
        lambda x: isinstance(x['node_feature'], tf.RaggedTensor), values)

    if sum(ragged) > 0:
        raise ValueError(
            'Found ragged tensors. Can only stack non-ragged tensors.')

    flat_sequence = tf.nest.map_structure(
        lambda x: tf.nest.flatten(x, expand_composites=True), values)
    
    dtype = values[0]['edge_src'].dtype

    flat_sequence = list(zip(*flat_sequence))

    flat_sequence_stacked = [
        fast_ragged_stack(x, dtype) for x in flat_sequence
    ]

    values = tf.nest.pack_sequence_as(
        structure, flat_sequence_stacked)

    del values['graph_indicator']

    return GraphTensor(values)


@tf.experimental.dispatch_for_api(tf.matmul)
def graph_tensor_matmul(
    a: GraphTensor,
    b,
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

def _mask_nodes(
    tensor: GraphTensor, 
    node_mask: tf.Tensor
) -> GraphTensor:
    
    num_nodes = tf.shape(tensor.node_feature)[0]

    # indices of nodes to keep
    keep_nodes = tf.boolean_mask(tf.range(num_nodes), node_mask)
    
    # Get edge mask: 
    # edges where edge_dst AND edge_src exist in `keep_nodes` will be kept
    edge_mask = tf.logical_and(
        tf.reduce_any(tf.expand_dims(tensor.edge_src, -1) == keep_nodes, -1),
        tf.reduce_any(tf.expand_dims(tensor.edge_dst, -1) == keep_nodes, -1)
    )
    
    # Decrement (node) indices in edge_dst and edge_src:
    # as nodes are completely dropped, indices needs to be 
    # decremented accordingly.
    decr = tf.concat([[-1], keep_nodes], axis=0)
    decr = tf.math.cumsum(decr[1:] - decr[:-1] - 1)
    decr = tf.tensor_scatter_nd_add(
        tensor=tf.zeros((num_nodes,), dtype=decr.dtype),
        indices=tf.expand_dims(keep_nodes, -1),
        updates=decr)
    
    # Apply mask and decrement to edges
    edge_src = tf.boolean_mask(tensor.edge_src, edge_mask)
    edge_src -= tf.gather(decr, edge_src)
    edge_dst = tf.boolean_mask(tensor.edge_dst, edge_mask)
    edge_dst -= tf.gather(decr, edge_dst)
    
    # Obtain data of the GraphTensor
    data = tensor._data.copy()
    
    # Add new (masked) edge_dst and edge_src
    data['edge_src'] = edge_src
    data['edge_dst'] = edge_dst
    
    # Apply masks on the rest of the data and add to data dict
    # Both data associated with edges and nodes will be masked
    for key in data.keys():
        if key not in ['edge_src', 'edge_dst']: # if not yet masked
            if key.startswith('edge'):
                 data[key] = tf.boolean_mask(data[key], edge_mask)
            else:
                 data[key] = tf.boolean_mask(data[key], node_mask)  
                
    return GraphTensor(**data)

def _mask_edges(
    tensor: GraphTensor, 
    edge_mask: tf.Tensor
) -> GraphTensor:
    # Obtain data of the GraphTensor
    data = tensor._data.copy()
    # Mask all data associated with edges
    for key in data.keys():
        if key.startswith('edge'):
             data[key] = tf.boolean_mask(data[key], edge_mask)   
    return GraphTensor(**data)

# TODO: Allow tf.boolean_mask(graph_tensor, mask) to mask graphs,
#       by specifying axis='graph'?
@tf.experimental.dispatch_for_api(tf.boolean_mask)
def graph_tensor_boolean_mask(
    tensor: GraphTensor, mask, axis=None,
) -> GraphTensor:
    '''Allows GraphTensor to be masked, via tf.boolean_mask.
    
    Conventiently, nodes or edges can be masked from the graph.

    Args:
        tensor (GraphTensor):
            An instance of a GraphTensor to be masked.
        mask (tf.Tensor):
            The node or edge mask. If the `mask` should be applied to
            nodes, it should match the outermost dim of `tensor.node_feature`;
            likewise if mask should be applied to edges, it should match the 
            outermost dim of `tensor.edge_src` and `tensor.edge_dst`.
        axis (int, str, None):
            If axis is set to None, 0, or 'node', nodes will be masked;
            otherwise, edges will be masked. `axis` usually does not accept
            strings; however, as (1) the axis to perform node and edge
            masking is always 0 anyways, and (2) additional arguments 
            could not be added, it was decided to use the `axis` argument
            to indicate whether nodes or edges should be masked.

    Returns:
        GraphTensor: Masked instance of a GraphTensor.

    '''
    if isinstance(tensor.node_feature, tf.RaggedTensor):
        ragged = True
        tensor = tensor.merge()
    else:
        ragged = False
    if isinstance(mask, tf.RaggedTensor):
        mask = mask.flat_values
    if 'node' in axis or not axis:
        tensor = _mask_nodes(tensor, mask)
    else:
        tensor = _mask_edges(tensor, mask)
    if ragged:
        return tensor.separate()
    return tensor

@tf.experimental.dispatch_for_unary_elementwise_apis(GraphTensor)
def graph_tensor_unary_elementwise_op_handler(api_func, x):
    '''Allows all unary elementwise operations (such as `tf.math.abs`)
    to handle graph tensors.
    '''
    return x.update({'node_feature': api_func(x.node_feature)})

@tf.experimental.dispatch_for_binary_elementwise_apis(
    Union[GraphTensor, tf.Tensor, float], Union[GraphTensor, tf.Tensor, float]
)
def graph_tensor_binary_elementwise_op_handler(api_func, x, y):
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



