import tensorflow as tf
import numpy as np
import typing
import warnings

from collections.abc import Iterable

from molgraph.layers import gnn_ops


BatchableExtensionType = tf.experimental.BatchableExtensionType
ExtensionTypeBatchEncoder = tf.experimental.ExtensionTypeBatchEncoder

dispatch_for_api = tf.experimental.dispatch_for_api
dispatch_for_unary_elementwise_apis = tf.experimental.dispatch_for_unary_elementwise_apis
dispatch_for_binary_elementwise_apis = tf.experimental.dispatch_for_binary_elementwise_apis


DEFAULT_FIELDS = [ 
    'sizes',
    'node_feature', 
    'edge_src', 
    'edge_dst', 
    'edge_feature', 
    'edge_weight', 
    'node_position',
]

DEFAULT_NODE_FIELDS = [
    'node_feature', 
    'node_position',
]

DEFAULT_EDGE_FIELDS = [
    'edge_src', 
    'edge_dst', 
    'edge_feature', 
    'edge_weight',
]

REQUIRED_FIELDS = [
    'sizes',
    'node_feature',
    'edge_src',
    'edge_dst',
]

NON_UPDATABLE_FIELDS = [
    'sizes',
    'edge_src',
    'edge_dst',
]


DEFAULT_FEATURE_DTYPE = tf.float32
DEFAULT_INDEX_DTYPE = tf.int64

Tensor = typing.Union[tf.Tensor, tf.RaggedTensor]
TensorSpec = typing.Union[tf.TensorSpec, tf.RaggedTensorSpec]
TensorOrTensorSpec = typing.Union[Tensor, TensorSpec]
TensorSpecOrShape = typing.Union[TensorSpec, tf.TensorShape]


class GraphBatchEncoder(ExtensionTypeBatchEncoder):

    '''A custom batch encoder for the :class:`~GraphTensor` class.

    This custom batch encoder allows a :class:`~GraphTensor` instance to be
    batched even when it is in its non-ragged state (namely encoding a
    disjoint graph). 

    In other words, a :class:`~GraphTensor` instance can be used seamlessly 
    with `tf.data.Dataset` as well as `tf.keras.Model`'s `fit`, `predict` and 
    `evaluate`.

    Example usage:

    >>> # Obtain a GraphTensor instance encoding a disjoint graph
    >>> graph_tensor = molgraph.GraphTensor(
    ...     sizes=[2, 3],
    ...     node_feature=[
    ...         [1.0, 0.0],
    ...         [1.0, 0.0],
    ...         [1.0, 0.0],
    ...         [1.0, 0.0],
    ...         [0.0, 1.0]
    ...     ],
    ...     node_state=[1., 2., 3., 4., 5.],
    ...     edge_src=[1, 0, 3, 4, 2, 4, 3, 2],
    ...     edge_dst=[0, 1, 2, 2, 3, 3, 4, 4],
    ... )
    >>> # Create a complicated dataset pipeline
    >>> ds = tf.data.Dataset.from_tensor_slices(graph_tensor)
    >>> ds = ds.map(lambda x: x).batch(2).unbatch().batch(2).map(lambda x: x)
    >>> for x in ds:
    ...     pass

    '''
    
    def batch(
        self, 
        spec: 'GraphTensor.Spec', 
        batch_size: typing.Optional[int]
    ) -> 'GraphTensor.Spec':

        '''Batches the :class:`~GraphTensor.Spec` instance.
        
        If the original spec corresponds to a non-ragged :class:`~GraphTensor`
        instance, then the resulting spec should always be a `tf.TensorSpec`,
        otherwise always a `tf.RaggedTensorSpec`.

        Args:
            spec (GraphTensor.Spec):
                The spec of the :class:`~GraphTensor` instance.
            batch_size (int, None):
                The batch size.
        Returns:
            A new :class:`~GraphTensor.Spec` instance with nested specs
            batched.
        '''
        
        def batch_field(x: TensorSpec) -> TensorSpec:

            if isinstance(x, tf.TensorSpec):
                # Non-ragged GraphTensor
                return tf.TensorSpec(
                    shape=[None] + x.shape[1:],
                    dtype=x.dtype)
            else:
                # Ragged GraphTensor
                return tf.RaggedTensorSpec(
                    shape=[batch_size, None] + x.shape[1:],
                    dtype=x.dtype,
                    ragged_rank=1,
                    row_splits_dtype=x.row_splits_dtype)
            
        batched_data_spec = tf.nest.map_structure(batch_field, spec.data_spec)
        sizes = tf.TensorSpec(shape=(None,), dtype=spec.sizes.dtype)
        return spec.__class__(sizes=sizes, **batched_data_spec)
    
    def unbatch(self, spec: 'GraphTensor.Spec') -> 'GraphTensor.Spec':

        '''Unbatches the :class:`~GraphTensor.Spec` instance.
        
        If the original spec corresponds to a non-ragged :class:`~GraphTensor`
        instance, then the resulting spec should always be a `tf.TensorSpec`,
        otherwise always a `tf.RaggedTensorSpec`.

        Args:
            spec (GraphTensor.Spec):
                The spec of the :class:`~GraphTensor` instance.

        Returns:
            A new :class:`~GraphTensor.Spec` instance with nested specs
            unbatched.
        '''
        
        def unbatch_field(x: TensorSpec):
            
            if isinstance(x, tf.TensorSpec):
                # Non-ragged GraphTensor
                return tf.TensorSpec(
                    shape=[None] + x.shape[1:],
                    dtype=x.dtype)
            else:
                # Ragged GraphTensor
                return tf.RaggedTensorSpec(
                    shape=[None] + x.shape[2:],
                    dtype=x.dtype,
                    ragged_rank=0,
                    row_splits_dtype=x.row_splits_dtype)
    
        unbatched_data_spec = tf.nest.map_structure(unbatch_field, spec.data_spec)
        sizes = tf.TensorSpec(shape=(), dtype=spec.sizes.dtype)
        return spec.__class__(sizes=sizes, **unbatched_data_spec)
        
    def encode(
        self, 
        spec: 'GraphTensor.Spec', 
        value: 'GraphTensor', 
        minimum_rank: int = 0
    ) -> typing.List[Tensor]:
        
        '''Encodes the GraphTensor instance.
        
        Should return a list of data values from the :class:`~GraphTensor` 
        instance, namely all its nested data values as a list. 

        Args:
            spec (GraphTensor.Spec):
                The spec of the :class:`~GraphTensor` instance.
            value (GraphTensor):
                The :class:`~GraphTensor` instance. 
            minimum_rank (int):
                Not used. 

        Returns:
            List of tensors (tf.RaggedTensor values or tf.Tensor values) 
            associated with the GraphTensor instance.
        '''

        separate = False if (spec.is_ragged() or spec.is_scalar()) else True 

        if separate:
            value = value.separate()

        # No need to pass along `sizes` as it can be inferred:
        # encoded graph tensor is either ragged or scalar
        return list(value.data.values())
    
    def encoding_specs(
        self, 
        spec: 'GraphTensor.Spec'
    ) -> typing.List[TensorSpec]:
 
        '''Encodes the spec of the GraphTensor instance.
        
        Should return a list of data specs from the :class:`~GraphTensor.Spec` 
        instance, namely all its nested data specs as a list. 

        The data specs returned by this method should correspond to the 
        data values returned by :meth:`~GraphBatchEncoder.encode`. 

        Args:
            spec (GraphTensor.Spec):
                The spec of the :class:`~GraphTensor` instance.

        Returns:
            List of specs (tf.RaggedTensorSpec specs or tf.TensorSpec specs) 
            associated with the GraphTensor.Spec instance, and corresponds
            to the list of values returned by :meth:`~GraphBatchEncoder.encode`. 
        '''

        ragged_rank, batch_shape = (
            (0, [None]) if spec.is_scalar() else (1, [None, None]))
        
        row_splits_dtype = spec.edge_src.dtype

        def encode_fields(x: TensorSpec) -> TensorSpec:

            if isinstance(x, tf.RaggedTensorSpec):
                return x
            else:
                return tf.RaggedTensorSpec(
                    shape=batch_shape + x.shape[1:], 
                    dtype=x.dtype, 
                    ragged_rank=ragged_rank,
                    row_splits_dtype=row_splits_dtype)

        # No need to pass along `sizes` as it can be inferred:
        # encoded graph tensor is either ragged or scalar
        encoded_data_spec = tf.nest.map_structure(encode_fields, spec.data_spec)
        return list(encoded_data_spec.values())
    
    def decode(
        self, 
        spec: 'GraphTensor.Spec', 
        encoded_value: typing.List[Tensor]
    ) -> 'GraphTensor':

        '''Decodes the encoded list of values.
        
        Should return a :class:`~GraphTensor` instance from the encoded values.

        This method reverses :meth:`~GraphBatchEncoder.encode`. 
   
        Args:
            spec (GraphTensor.Spec):
                The original spec of the original :class:`~GraphTensor` instance.
                The spec is used to decode the encoded values.
            encoded_value (list[tf.Tensor], list[tf.RaggedTensor]):
                The encoded values of :meth:`~GraphBatchEncoder.encode`. 

        Returns:
            A new :class:`~GraphTensor` instance constructed from the encoded
            values. 
        '''

        graph_tensor = GraphTensor(**dict(zip(spec.data_spec, encoded_value)))

        if graph_tensor.is_ragged() and not spec.is_ragged():
            return graph_tensor.merge()

        return graph_tensor


class GraphTensor(BatchableExtensionType):
    
    '''A custom tensor encoding a graph.
    
    A (molecular) graph, encoded as a :class:`~GraphTensor` instance,
    could encode a single subgraph (single molecule) or multiple subgraphs 
    (multiple molecules). Furthermore, the :class:`~GraphTensor` 
    can either encode multiple molecules (molecular graphs) as a single 
    disjoint graph (nested `tf.Tensor` values) or multiple subgraphs 
    (nested `tf.RaggedTensor` values). It is recommended to encode a 
    (molecular) graph as a disjoint graph as it is a significantly more
    efficient representation, both in terms of memory and runtime.

    Note: every method that seemingly modifies the :class:`~GraphTensor` 
    instance actually does not modify it. Instead, a new :class:`~GraphTensor` 
    instance is constructed and returned by these methods. This is necessary 
    to allow TF to properly track the :class:`~GraphTensor` instances. 
    These methods include: :meth:`~propagate`, :meth:`~merge`, 
    :meth:`~separate`, :meth:`~update`, :meth:`~remove`, etc.

    Args:
        sizes (tf.Tensor):
            A 1-D or 0-D tf.Tensor specifying the sizes of the subgraphs.
        node_feature (tf.Tensor, tf.RaggedTensor):
            A 2-D tf.Tensor or 3-D tf.RaggedTensor encoding the features
            associated with the nodes of the graph.
        edge_src (tf.Tensor, tf.RaggedTensor):
            A 1-D tf.Tensor or 2-D tf.RaggedTensor encoding the source node 
            indices of the edges of the graph. Entry i in edge_src 
            corresponds to node i (index i of node_Feature).
        edge_dst (tf.Tensor, tf.RaggedTensor):
            A 1-D tf.Tensor or 2-D tf.RaggedTensor encoding the destination 
            (target) node indices of the edges of the graph. Entry i in edge_src 
            corresponds to node i (index i of node_Feature).
        edge_feature (tf.Tensor, tf.RaggedTensor, None):
            A 2-D tf.Tensor or 3-D tf.RaggedTensor encoding the features 
            associated with the edges of the graph. Index j corresponds to 
            edge j (index j of edge_src and edge_dst). Edge features are 
            optional, but commonly used for molecular graphs.
        edge_weight (tf.Tensor, tf.RaggedTensor, None):
            A 1-D tf.Tensor or 2-D tf.RaggedTensor encoding the weights 
            associated with the edges of the graph. Index j corresponds
            to edge j (index j of edge_feature, edge_src and edge_dst). Edge 
            weights are optional, but useful to encode e.g. attention 
            coefficients.
        node_position (tf.Tensor, tf.RaggedTensor, None):
            A 2-D tf.Tensor or 3-D tf.RaggedTensor encoding the node positions 
            (commony laplacian positional encoding) corresponding to the nodes. 
            Index i corresponds to node i (index i of node_feature). Node 
            positions are optional, but useful to better encode <3D molecular 
            graphs wherein node positions are not encoded.
        **auxiliary (tf.Tensor, tf.RaggedTensor):
            Auxiliary graph data to be supplied to the :class:`~GraphTensor` 
            instance. These are user specified data fields and can be useful
            to supplement the graph with additional information. If the 
            data field added should be associated with the edges or nodes of 
            the graph, prepend 'edge' or 'node' to the names respectively.
            If not, a single underscore ('_') needs to be prepended; an 
            underscore indiates that the field is static and should not be
            manipulated (e.g. with :meth:`~merge`, :meth:`~separate`). A static
            field should not be used in a `tf.data.Dataset` instance as it
            requires the data fields to be non-static.

    Example usage:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     sizes=[2, 3],
    ...     node_feature=[
    ...         [1.0, 0.0],
    ...         [1.0, 0.0],
    ...         [1.0, 0.0],
    ...         [1.0, 0.0],
    ...         [0.0, 1.0]
    ...     ],
    ...     edge_src=[1, 0, 3, 4, 2, 4, 3, 2],
    ...     edge_dst=[0, 1, 2, 2, 3, 3, 4, 4],
    ... )
    >>> gnn_model = tf.keras.Sequential([
    ...     molgraph.layers.GCNConv(32),
    ...     molgraph.layers.GCNConv(32)
    ... ])
    >>> gnn_model.predict(graph_tensor, verbose=0).shape
    TensorShape([2, None, 32])
    '''

    __name__ = 'molgraph.tensors.GraphTensor'
    
    sizes: tf.Tensor
    node_feature: Tensor
    edge_src: Tensor
    edge_dst: Tensor
    edge_feature: typing.Optional[Tensor]
    edge_weight: typing.Optional[Tensor]
    node_position: typing.Optional[Tensor]

    auxiliary: typing.Mapping[str, Tensor]
        
    __batch_encoder__ = GraphBatchEncoder()
    
    def __init__(
        self,
        sizes: typing.Optional[tf.Tensor] = None,
        node_feature: Tensor = None,
        edge_src: Tensor = None,  
        edge_dst: Tensor = None,  
        edge_feature: typing.Optional[Tensor] = None,
        edge_weight: typing.Optional[Tensor] = None,
        node_position: typing.Optional[Tensor] = None,
        **auxiliary: Tensor
    ) -> None:

        assert node_feature is not None, ('`node_feature` is a required field.')
        assert edge_src is not None, ('`edge_src` is a required field.')
        assert edge_dst is not None, ('`edge_dst` is a required field.')

        node_feature = _maybe_convert_to_tensor(node_feature)
        edge_src = _maybe_convert_to_tensor(edge_src)
        edge_dst = _maybe_convert_to_tensor(edge_dst)
        sizes = _maybe_convert_to_tensor(sizes)

        if sizes is None:
            if hasattr(node_feature, 'row_lengths'):
                # sizes can be inferred if ragged GraphTensor
                sizes = tf.cast(
                    node_feature.row_lengths(), dtype=edge_src.dtype)
            else:
                sizes = tf.shape(
                    node_feature, out_type=edge_src.dtype)[0]
                
        self.sizes = sizes
        self.node_feature = node_feature
        self.edge_src = edge_src
        self.edge_dst = edge_dst
        self.edge_feature = _maybe_convert_to_tensor(edge_feature)
        self.edge_weight = _maybe_convert_to_tensor(edge_weight)
        self.node_position = _maybe_convert_to_tensor(node_position)

        self.auxiliary = {
            key: _maybe_convert_to_tensor(value) 
            for (key, value) in auxiliary.items()}
        
    def __validate__(self) -> None:

        '''Validates the newly instatiated GraphTensor instance.
        
        To simplify the validation, and to avoid overhead when running
        in graph mode, validation is only performed in eager mode.
        '''

        if tf.executing_eagerly():
            _check_compatible_types(self)
            _check_ranks(self)
            _check_compatible_sizes(self)
            _check_edges(self)
            
    def update(
        self, 
        data: typing.Optional[typing.Mapping[str, Tensor]] = None, 
        **data_as_kwargs: Tensor
    ) -> 'GraphTensor':
        
        '''Update data field(s) of the :class:`~GraphTensor` instance.

        This method either updates existing data fields or adds 
        new data fields to the :class:`~GraphTensor` instance.

        Caution when adding new data fields: 
            *   If name of data field starts with 'node' or 'edge' it is 
                assumed that the size of the corresponding values match with
                that of `node_feature` or `edge_src` respetively. In other words,
                the new data need to encode the same number of nodes or 
                edges respectively.
            *   If new data should not be associated with the nodes or
                edges of the :class:`~GraphTensor` instance, then the name 
                of the data field should start with and underscore ('_'). 
                The underscore indicate that the corresponding values are 
                static and should not be tampered with. 
        
        Caution when updating the :class:`~GraphTensor` instance with values
        of a different type. E.g. when updating a :class:`~GraphTensor` instance
        (encoding nested `tf.RaggedTensor` values) with `tf.Tensor` values:
    
            *   A :class:`~GraphTensor` instance should only be updated with
                values originating from the existing values, or corresponding
                to the existing values. Reason: although very  rare, tf.Tensor 
                values coming from another graph structure may have the same 
                size (namely, the same node or edge dimension), but different 
                row lengths (namely, different sized subgraphs). This will 
                result in a silent error, where the :class:`~GraphTensor` 
                instance is updated without error, but with wrongly partioned values.

        Example usage:

        >>> graph_tensor = molgraph.GraphTensor(
        ...     sizes=[2, 3],
        ...     node_feature=[
        ...         [1.0, 0.0],
        ...         [1.0, 0.0],
        ...         [1.0, 0.0],
        ...         [1.0, 0.0],
        ...         [0.0, 1.0]
        ...     ],
        ...     edge_src=[1, 0, 3, 4, 2, 4, 3, 2],
        ...     edge_dst=[0, 1, 2, 2, 3, 3, 4, 4],
        ... )
        >>> random_node_features = tf.random.uniform(
        ...     graph_tensor.node_feature.shape
        ... )
        >>> random_edge_features = tf.random.uniform(
        ...     graph_tensor.edge_src.shape.concatenate([1])
        ... )
        >>> graph_tensor = graph_tensor.update({
        ...     'node_feature': random_node_features, 
        ...     'edge_feature': random_edge_features,
        ... })

        Args:
            data (dict):
                Nested data. Specifically, a dictionary of tensors.

        Returns:
            A new updated :class:`~GraphTensor` instance.
        '''

        def convert_value(key, new_value: Tensor, old_value: Tensor) -> Tensor:
            
            if (
                isinstance(new_value, tf.RaggedTensor) and
                isinstance(old_value, tf.Tensor)
            ):
                new_value = new_value.flat_values
                _assert_compatible_outer_shape(key, old_value, new_value)
            elif (
                isinstance(new_value, tf.Tensor) and  
                isinstance(old_value, tf.RaggedTensor)
            ):
                new_value = old_value.with_flat_values(new_value)
            else:
                _assert_compatible_outer_shape(key, old_value, new_value)
            return new_value
    
        if data is None:
            data = {}

        data.update(data_as_kwargs)

        data = tf.nest.map_structure(_maybe_convert_to_tensor, data)

        existing_data = self.data 

        for key, value in data.items():
            if key in NON_UPDATABLE_FIELDS:
                raise ValueError(
                    f'Currently, data field {key} cannot be updated. '
                     'A workaround is to instantiate a GraphTensor instance '
                     'from its constructor.' )
            if key in existing_data:
                data[key] = convert_value(
                    key, value, existing_data[key])
            elif key.startswith('node'):
                data[key] = convert_value(
                    key, value, existing_data['node_feature'])
            elif key.startswith('edge'):
                data[key] = convert_value(
                    key, value, existing_data['edge_src'])
            elif key.startswith('_'):
                data[key] = value
            else:
                raise ValueError(
                    f'Data field {key} not recognized. For user specified data '
                     'fields, either prepend `node` or `edge` to the name, '
                     'or `_` to indicate a static data field which will just '
                     'be passed along with the GraphTensor instance as is.')
            
        existing_data.update(data)
        return self.__class__(sizes=self.sizes, **existing_data)
        
    def remove(
        self, 
        fields: typing.Union[str, typing.List[str]]
    ) -> 'GraphTensor':
        
        '''Removes data from the :class:`~GraphTensor` instance.

        Example usage:
        
        >>> graph_tensor = molgraph.GraphTensor(
        ...     sizes=[2, 3],
        ...     node_feature=[
        ...         [1.0, 0.0],
        ...         [1.0, 0.0],
        ...         [1.0, 0.0],
        ...         [1.0, 0.0],
        ...         [0.0, 1.0]
        ...     ],
        ...     edge_feature=[
        ...         [1.0, 0.0],
        ...         [1.0, 0.0],
        ...         [1.0, 0.0],
        ...         [1.0, 0.0],
        ...         [1.0, 0.0],
        ...         [1.0, 0.0],
        ...         [1.0, 0.0],
        ...         [1.0, 0.0],
        ...     ],
        ...     edge_src=[1, 0, 3, 4, 2, 4, 3, 2],
        ...     edge_dst=[0, 1, 2, 2, 3, 3, 4, 4],
        ... )
        >>> graph_tensor = graph_tensor.remove(['edge_feature'])

        Args:
            fields (str, list[str]):
                Data fields to be removed from the :class:`~GraphTensor` 
                instance. Currently, `edge_dst`, `edge_src`, `node_feature` 
                and `sizes` cannot be removed.

        Returns:
            GraphTensor: An updated :class:`~GraphTensor` instance.
        '''

        data = self.data
        if isinstance(fields, str):
            fields = [fields]
        for key in fields:
            if key in REQUIRED_FIELDS:
                raise ValueError(f'Data field {key} cannot be removed.')
            elif key in fields:
                del data[key]
        return self.__class__(sizes=self.sizes, **data)
        
    def separate(
        self, 
        other: typing.Optional['GraphTensor'] = None, /
    ) -> 'GraphTensor':
        
        '''Converts the :class:`~GraphTensor` into a ragged state. 

        In other words, this method separates each subgraph of the 
        :class:`~GraphTensor` instance, resulting in a new :class:`~GraphTensor`
        instance with each subgraph separated by rows:

        >>> graph_tensor = molgraph.GraphTensor(
        ...     sizes=[2, 3],
        ...     node_feature=[
        ...         [1.0, 0.0],
        ...         [1.0, 0.0],
        ...         [1.0, 0.0],
        ...         [1.0, 0.0],
        ...         [0.0, 1.0]
        ...     ],
        ...     edge_src=[1, 0, 3, 4, 2, 4, 3, 2],
        ...     edge_dst=[0, 1, 2, 2, 3, 3, 4, 4],
        ... )
        >>> graph_tensor = graph_tensor.separate()

        This method can optionally be used as a "static method" to separate
        another :class:`~GraphTensor` instance:

        >>> graph_tensor = molgraph.GraphTensor(
        ...     sizes=[2, 3],
        ...     node_feature=[
        ...         [1.0, 0.0],
        ...         [1.0, 0.0],
        ...         [1.0, 0.0],
        ...         [1.0, 0.0],
        ...         [0.0, 1.0]
        ...     ],
        ...     edge_src=[1, 0, 3, 4, 2, 4, 3, 2],
        ...     edge_dst=[0, 1, 2, 2, 3, 3, 4, 4],
        ... )
        >>> ds = tf.data.Dataset.from_tensor_slices(graph_tensor)
        >>> ds = ds.batch(2).map(molgraph.GraphTensor.separate)

        Note: although not a common use case, the `separate` and `merge` 
        methods are both implemented in this way to make it convenient to go 
        between states with `tf.data.Dataset`. 
        
        Args:
            other (None, GraphTensor):
                A :class:`~GraphTensor` instance passed as a 
                positional-argument-only. If None, `self` will be separated. 
                Default to None.

        Returns:
            GraphTensor: A :class:`~GraphTensor` instance with its subgraphs
            separated into rows (nested ragged tensors).
        '''
           
        obj = self if other is None else other
        
        if obj.is_ragged():
            raise ValueError(f'{obj} is already in its ragged state.')

        _assert_no_intersubgraph_edges(obj)

        data = obj.data 

        if self.is_scalar():
            data = tf.nest.map_structure(
                lambda x: tf.RaggedTensor.from_row_starts(
                    x, tf.constant([0], dtype=obj.edge_src.dtype)), 
                data)
            return self.__class__(sizes=self.sizes, **data)
        
        graph_indicator_nodes = self.graph_indicator
        graph_indicator_edges = tf.gather(graph_indicator_nodes, data['edge_src'])

        num_subgraphs = obj.num_subgraphs

        for field, value in data.items():
            
            if field in DEFAULT_NODE_FIELDS:
                data[field] = tf.RaggedTensor.from_value_rowids(
                    value, graph_indicator_nodes, num_subgraphs)
                if field == 'node_feature':
                    edge_decrement = tf.gather(
                        data[field].row_starts(), graph_indicator_edges)
                    edge_decrement = tf.cast(
                        edge_decrement, dtype=data['edge_src'].dtype)
            elif field in DEFAULT_EDGE_FIELDS:
                if field in ['edge_src', 'edge_dst']:
                    value -= edge_decrement
                data[field] = tf.RaggedTensor.from_value_rowids(
                    value, graph_indicator_edges, num_subgraphs)
            elif field.startswith('node'):
                data[field] = tf.RaggedTensor.from_value_rowids(
                    value, graph_indicator_nodes, num_subgraphs)
            elif field.startswith('edge'):
                data[field] = tf.RaggedTensor.from_value_rowids(
                    value, graph_indicator_edges, num_subgraphs)
            elif field.startswith('_'):
                data[field] = value
            else:
                # Should not end up here, but raise error just in case.
                raise ValueError(
                    f'Data field {field} not recognized. For user specified data '
                     'fields, either prepend `node` or `edge` to the name, '
                     'or `_` to indicate a static data field which will just '
                     'be passed along with the GraphTensor instance as is.')

        return obj.__class__(sizes=self.sizes, **data)

    def merge(
        self, 
        other: typing.Optional['GraphTensor'] = None, /
    ) -> 'GraphTensor':
        
        '''Converts the :class:`~GraphTensor` into a non-ragged state.

        In other words, this method merged the row-separated subgraphs 
        into a single disjoint graph (all nodes and edges along the same
        dimension/row):
        
        >>> graph_tensor = molgraph.GraphTensor(
        ...     sizes=[2, 3],
        ...     node_feature=[
        ...         [1.0, 0.0],
        ...         [1.0, 0.0],
        ...         [1.0, 0.0],
        ...         [1.0, 0.0],
        ...         [0.0, 1.0]
        ...     ],
        ...     edge_src=[1, 0, 3, 4, 2, 4, 3, 2],
        ...     edge_dst=[0, 1, 2, 2, 3, 3, 4, 4],
        ... )
        >>> graph_tensor = graph_tensor.separate()
        >>> graph_tensor = graph_tensor.merge()

        This is the preferred state of a :class:`~GraphTensor`
        instance as it is an efficient representation. 

        This method can optionally be used as a "static method" to merge
        another :class:`~GraphTensor` instance:

        >>> graph_tensor = molgraph.GraphTensor(
        ...     sizes=[2, 3],
        ...     node_feature=[
        ...         [1.0, 0.0],
        ...         [1.0, 0.0],
        ...         [1.0, 0.0],
        ...         [1.0, 0.0],
        ...         [0.0, 1.0]
        ...     ],
        ...     edge_src=[1, 0, 3, 4, 2, 4, 3, 2],
        ...     edge_dst=[0, 1, 2, 2, 3, 3, 4, 4],
        ... )
        >>> graph_tensor = graph_tensor.separate()
        >>> ds = tf.data.Dataset.from_tensor_slices(graph_tensor)
        >>> ds = ds.batch(2).map(molgraph.GraphTensor.merge)

        Note: although not a common use case, the `separate` and `merge` 
        methods are both implemented in this way to make it convenient to go 
        between states with `tf.data.Dataset`. 
        
        Args:
            other (None, GraphTensor):
                A :class:`~GraphTensor` instance passed as a 
                positional-argument-only. If None, `self` will be merged. 
                Default to None.

        Returns:
            GraphTensor: A :class:`~GraphTensor` instance with its subgraphs
            merged into a single disjoint graph (nested "rectangular" tensors).
        '''

        obj = self if other is None else other
        
        if not obj.is_ragged():
            raise ValueError(f'{obj} is already in its non-ragged state.')
        
        data = obj.data

        edge_increment = tf.gather(
            data['node_feature'].row_starts(), data['edge_src'].value_rowids())
        edge_increment = tf.cast(edge_increment, dtype=data['edge_src'].dtype)

        for key, value in data.items():
            if not key.startswith('_'):
                data[key] = value.flat_values
    
        data['edge_src'] += edge_increment
        data['edge_dst'] += edge_increment
        return obj.__class__(sizes=self.sizes, **data)
    
    def propagate(
        self, 
        mode: typing.Optional[str] = 'sum',
        normalize: bool = False,
        reduction: typing.Optional[str] = None,
        residual: typing.Optional[tf.Tensor] = None,
        **kwargs,
    ) -> 'GraphTensor':
        
        # TODO: Move residual, activation and reduction out from this method?

        '''Propagates node features of the :class:`~GraphTensor` instance.

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
        
        Example usage:

        >>> graph_tensor = molgraph.GraphTensor(
        ...     sizes=[2, 3],
        ...     node_feature=[
        ...         [1.0, 0.0],
        ...         [1.0, 0.0],
        ...         [1.0, 0.0],
        ...         [1.0, 0.0],
        ...         [0.0, 1.0]
        ...     ],
        ...     edge_src=[1, 0, 3, 4, 2, 4, 3, 2],
        ...     edge_dst=[0, 1, 2, 2, 3, 3, 4, 4],
        ... )
        >>> graph_tensor = graph_tensor.propagate()

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
            data = self.merge().data
        else:
            data = self.data

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
            edge_weight=data.pop('edge_weight', None),
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
        
        graph_tensor = self.__class__(sizes=self.sizes, **data)

        if not self.is_ragged():
            return graph_tensor 
        
        return graph_tensor.separate()
    
    def is_scalar(
        self, 
        other: typing.Optional['GraphTensor'] = None, /
    ) -> bool:
        
        '''Checks whether the :class:`~GraphTensor` instance is a "scalar".

        A "scalar" is loosely defined, but basically means that the 
        :class:`~GraphTensor` instance is "unbatched". The method should
        rarely be used. Currently only used for the custom batch encoder 
        to flag that the :class:`~GraphTensor` instance should not be 
        separated when encoded (in the custom batch encoder).
        
        Returns:
            A boolean indicating whether the :class:`~GraphTensor` instance
            is a "scalar".
        '''

        obj = self if other is None else other
        return obj.sizes.shape.rank == 0

    def is_ragged(
        self, 
        other: typing.Optional['GraphTensor'] = None, /
    ) -> bool:
        
        '''Checks whether the :class:`~GraphTensor` instance is in its ragged state.
        
        Returns:
            A boolean indicating whether the :class:`~GraphTensor` instance
            is in its ragged state.
        '''

        obj = self if other is None else other
        return isinstance(obj.node_feature, tf.RaggedTensor)
    
    @property
    def num_subgraphs(self) -> tf.Tensor:

        '''The number of subgraphs the :class:`~GraphTensor` instance is encoding.
        
        Returns:
            An integer specifying the number of subgraphs.
        '''

        if self.is_scalar():
            num_subgraphs = tf.constant(1, dtype=self.sizes.dtype)
        else:
            num_subgraphs = tf.shape(self.sizes, out_type=self.sizes.dtype)[0]
        if tf.executing_eagerly():
            return num_subgraphs.numpy()
        return num_subgraphs
    
    @property
    def graph_indicator(self) -> typing.Union[tf.Tensor, None]:

        '''The graph indicator of the :class:`~GraphTensor` instance.
        
        The graph indicator indicates what graphs the nodes are 
        associated with. Only relevant when the :class:`~GraphTensor` instance
        is in its non-ragged state.

        Returns:
            In the non-ragged state, returns a tf.Tensor encoding graph
            indicator. Otherwise returns None.
        '''

        if self.is_ragged():
            return None 
        elif self.is_scalar():
            return tf.zeros(
                tf.shape(self.node_feature)[:1], dtype=self.sizes.dtype)
        return tf.repeat(
            tf.range(tf.shape(self.sizes, out_type=self.sizes.dtype)[0]), 
            self.sizes)
    
    @property
    def spec(self) -> 'GraphTensor.Spec':

        '''Spec of the :class:`~GraphTensor` instance.

        Unlike `_type_spec`, `spec` specifies a more realistic (or rather
        useful) specification of the :class:`~GraphTensor` instance. In 
        other words, the dimension corresponding to the size of the disjoint 
        graph (i.e. number of nodes and edges) is `None` rather than a 
        specific value. If exact specification is desired, use `_type_spec`
        or `tf.type_spec_from_value` instead.

        Returns:
            GraphTensor.Spec: the corresponding spec of the :class:`~GraphTensor`.
        '''

        def unspecify_size(x: TensorSpec) -> TensorSpec:

            if isinstance(x, tf.TensorSpec):
                return tf.TensorSpec(
                    shape=[None] + x.shape[1:],
                    dtype=x.dtype)
            else:
                return tf.RaggedTensorSpec(
                    shape=x.shape[:1] + [None] + x.shape[2:],
                    dtype=x.dtype,
                    ragged_rank=x.ragged_rank,
                    row_splits_dtype=x.row_splits_dtype)
        
        spec = tf.type_spec_from_value(self)
        data_spec = spec.data_spec 
        data_spec = tf.nest.map_structure(unspecify_size, data_spec)
        if spec.sizes.shape.rank == 0:
            sizes = spec.sizes
        elif spec.is_ragged():
            sizes = tf.TensorSpec(spec.node_feature.shape[:1], spec.sizes.dtype)
        else:
            sizes = tf.TensorSpec([None], spec.sizes.dtype)
        return GraphTensor.Spec(sizes=sizes, **data_spec)

    @property
    def shape(self) -> tf.TensorShape:

        '''Partial shape of the :class:`~GraphTensor` instance.
        
        Note: `shape` now returns a `tf.TensorShape` with the following 
        dimensions, regardless of its state: (num_subgraphs, num_nodes, num_features)

        Returns:
            tf.TensorShape: the partial shape of the :class:`~GraphTensor`.
        '''

        return self.sizes.shape.concatenate(
            tf.TensorShape([None]).concatenate(self.node_feature.shape[-1:]))
    
    @property
    def dtype(self) -> tf.DType:

        '''Partial dtype of the :class:`~GraphTensor` instance.

        Returns:
            tf.DType: the partial dtype of the :class:`~GraphTensor`.
        '''

        return self.node_feature.dtype
    
    @property
    def rank(self) -> int:

        '''Partial rank of the :class:`~GraphTensor` instance.
    
        Returns:
            int: the partial rank of the :class:`~GraphTensor`.
        '''

        return self.shape.rank
    
    @property
    def data(self) -> typing.Mapping[str, Tensor]:

        '''Unpacks the nested data of the :class:`~GraphTensor` instance.

        meth:`~data` corresponds to :meth:`~GraphTensor.Spec.data_spec`. 

        When working with values returned from `data`, make sure to 
        also work with specs returned from `data_spec` of the associated 
        :class:`~GraphTensor.Spec` instance. If not, conflicts will likely occur.

        This property is implemented to be more selective and have more 
        control over what data should be unpacked from the 
        :class:`~GraphTensor` instance. 

        Unfortunately, `tf.nest` ops cannot be used directly on the 
        :class:`~GraphTensor` instance as it will expand all composites 
        including `tf.RaggedTensor` values. E.g., when performing
        `tf.nest.map_structure` it will flatten `tf.RaggedTensor` into its
        composites, which is an undesired behavior. By returning a dict 
        of nested data, `tf.nest` ops can be used on the dict without specifying
        expand_composites=True, resulting in tf.RaggedTensor values not being 
        flattened. Furthermore, it allows us to unpack the `auxiliary` data
        approprietly. 
    
        Returns:
            A dictionary with nested data values.
        '''

        return _get_data_or_data_spec(self)
    
    def __getattr__(self, name: str) -> Tensor:

        '''Access data fields of the :class:`~GraphTensor` as attributes.

        Only called when attribute lookup has not found attribute `name` in 
        the usual places. 

        Args:
            name (str):
                The data field to be extracted.

        Returns:
            A tf.Tensor corresponding to the data field `name`. 
        
        Raises:
            AttributeError: if `name` is not a data field of 
            the :class:`~GraphTensor`.
        '''

        if name in self.__dict__:
            return self.__dict__[name]
            
        if name in object.__getattribute__(self, 'auxiliary'):
            return object.__getattribute__(self, 'auxiliary')[name]
        raise AttributeError(f'{name!r} not found.')
    
    def __getitem__(
        self,
        index: typing.Union[slice, int, typing.List[int]]
    ) -> typing.Union[tf.RaggedTensor, tf.Tensor, 'GraphTensor']:
        
        '''Access subgraphs of the :class:`~GraphTensor` via indexing.

        Args:
            index (slice, int, list[int]):
                Indices or slice for accessing certain subgraphs of the
                :class:`~GraphTensor` instance.

        Returns:
            A :class:`~GraphTensor` instance with the specified subgraphs.
        
        Raises:
            KeyError: if `index` (str) does not exist in data spec.
            tf.errors.InvalidArgumentError: if `index` (int, list[int]) is out 
            of range.
        '''
        
        if isinstance(index, slice):
            index = _slice_to_tensor(index, self.num_subgraphs)

        return tf.gather(self, index)

    def __repr__(self) -> str:
        
        '''A string representation of the :class:`~GraphTensor` instance.

        Compared to the default `__repr__`, this `__repr__` is less verbose. 
        '''

        def to_string(item: typing.Tuple[str, Tensor]) -> str:
            key, value = item
            if isinstance(value, tf.Tensor):
                return (
                    f'{key}=<tf.Tensor: '
                    f'shape={value.shape}, '
                    f'dtype={value.dtype.name}>')
            return (
                f'{key}=<tf.RaggedTensor: '
                f'shape={value.shape}, '
                f'dtype={value.dtype.name}, '
                f'ragged_rank={value.ragged_rank}>')
        
        fields = [(
            f'sizes=<tf.Tensor: shape={self.sizes.shape}, '
            f'dtype={self.sizes.dtype.name}>')]
        fields.extend([to_string(item) for item in self.data.items()])
        
        return f'GraphTensor(\n  ' + ',\n  '.join(fields) + ')'
    
    def __iter__(self) -> '_GraphTensorIterator':

        '''A :class:`~GraphTensor` iterable. 
        
        The implementaton of `__iter__` makes the :class:`~GraphTensor` 
        instance iterable, however, only in eager mode.
        '''

        if not tf.executing_eagerly():
            raise ValueError(
                'Can only iterate over a `GraphTensor` instance in eager mode.')
        return _GraphTensorIterator(self, limit=self.num_subgraphs)

    class Spec:

        # TODO: Add documentation regarding tf.TensorShape as input.

        '''The spec associated with a :class:`~GraphTensor` instance.

        Example usage:

        Obtain spec from existing GraphTensor instance:

        >>> graph_tensor = molgraph.GraphTensor(
        ...     sizes=[2, 3],
        ...     node_feature=[
        ...         [1.0, 0.0],
        ...         [1.0, 0.0],
        ...         [1.0, 0.0],
        ...         [1.0, 0.0],
        ...         [0.0, 1.0]
        ...     ],
        ...     edge_src=[1, 0, 3, 4, 2, 4, 3, 2],
        ...     edge_dst=[0, 1, 2, 2, 3, 3, 4, 4],
        ... )
        >>> # Two options, one which unspecifies node/edge dimension:
        >>> spec = graph_tensor.spec
        >>> # and one which specifies node/edge dimension:
        >>> spec = tf.type_spec_from_value(graph_tensor)

        Obtain spec from the class:

        >>> spec = GraphTensor.Spec(
        ...     sizes=tf.TensorSpec([None], tf.int64)
        ...     node_feature=tf.TensorSpec([None, 2], tf.float32),
        ...     edge_src=tf.TensorSpec([None], tf.int64),
        ...     edge_dst=tf.TensorSpec([None], tf.int64)
        ... )

        Args:
            sizes (tf.TensorSpec):
                `sizes` spec corresponding to `sizes` of 
                the associated :class:`~GraphTensor` instance.
            node_feature (tf.TensorSpec, tf.RaggedTensorSpec):
                `node_feature` spec corresponding to `node_feature` of 
                the associated :class:`~GraphTensor` instance.
            edge_src (tf.TensorSpec, tf.RaggedTensorSpec):
                `edge_src` spec corresponding to `edge_src` of 
                the associated :class:`~GraphTensor` instance.
            edge_dst (tf.TensorSpec, tf.RaggedTensorSpec):
                `edge_dst` spec corresponding to `edge_dst` of 
                the associated :class:`~GraphTensor` instance.
            edge_feature (tf.TensorSpec, tf.RaggedTensorSpec, None):
                `edge_feature`spec corresponding to `edge_feature` of 
                the associated :class:`~GraphTensor` instance.
            edge_weight (tf.TensorSpec, tf.RaggedTensorSpec, None):
                `edge_weight`spec corresponding to `edge_weight` of 
                the associated :class:`~GraphTensor` instance.
            node_position (tf.TensorSpec, tf.RaggedTensorSpec, None):
                `node_position` spec corresponding to `node_position` of 
                the associated :class:`~GraphTensor` instance.
            **auxiliary (tf.TensorSpec, tf.RaggedTensorSpec):
                Auxiliary graph data spec.
        '''           

        def __init__(
            self,
            sizes: typing.Optional[TensorSpecOrShape] = None,
            node_feature: typing.Optional[TensorSpecOrShape] = None,
            edge_src: typing.Optional[TensorSpecOrShape] = None,
            edge_dst: typing.Optional[TensorSpecOrShape] = None,
            edge_feature: typing.Optional[TensorSpecOrShape] = None,
            edge_weight: typing.Optional[TensorSpecOrShape] = None,
            node_position: typing.Optional[TensorSpecOrShape] = None,
            **auxiliary: TensorSpecOrShape,
        ) -> None:

            node_feature = _get_spec(node_feature, DEFAULT_FEATURE_DTYPE, 2)
            force_shape = (
                node_feature.shape[:1] 
                if isinstance(node_feature, tf.TensorSpec) 
                else node_feature.shape[:2])
            edge_src = _get_spec(edge_src, DEFAULT_INDEX_DTYPE, 1, force_shape)
            edge_dst = _get_spec(edge_dst, DEFAULT_INDEX_DTYPE, 1, force_shape)

            sizes_dtype = (
                edge_src.dtype if sizes is None else sizes.dtype)
            if sizes is None or sizes.shape.rank > 0:
                self.sizes = tf.TensorSpec([None], sizes_dtype)
            else:
                self.sizes = tf.TensorSpec([], sizes_dtype)

            self.node_feature = node_feature
            self.edge_src = edge_src
            self.edge_dst = edge_dst
            self.edge_feature = _get_spec(edge_feature, DEFAULT_FEATURE_DTYPE, 2)
            self.edge_weight = _get_spec(edge_weight, DEFAULT_FEATURE_DTYPE, 1)
            self.node_position = _get_spec(node_position, DEFAULT_FEATURE_DTYPE, 2)

            self.auxiliary = {
                k: _get_spec(v, DEFAULT_FEATURE_DTYPE, 2) for (k, v) in auxiliary.items()}

        @property
        def shape(self) -> tf.TensorShape:

            '''Partial shape of the :class:`~GraphTensor.Spec` instance.
            
            Note: `shape` now returns a `tf.TensorShape` with the following 
            dimensions, regardless of its state: (num_subgraphs, num_nodes[ragged]).
            As of now, num_subgraphs cannot be obtained when nested data specs 
            are tf.TensorSpec specs.

            Returns:
                tf.TensorShape: the partial shape of the :class:`~GraphTensor.Spec`.
            '''

            return self.sizes.shape.concatenate(
                tf.TensorShape([None]).concatenate(self.node_feature.shape[-1:]))
     
        
        @property
        def dtype(self) -> tf.DType:

            '''Partial dtype of the :class:`~GraphTensor.Spec` instance.
            
            Returns:
                tf.DType: the partial dtype of the :class:`~GraphTensor.Spec`.
            '''

            return self.node_feature.dtype
        
        @property
        def rank(self) -> int:

            '''Partial rank of the :class:`~GraphTensor.Spec` instance.
            
            Returns:
                int: the partial rank of the :class:`~GraphTensor.Spec`.
            '''

            return self.shape.rank
    
        @property
        def data_spec(self) -> typing.Mapping[str, TensorSpec]:

            '''Unpacks the nested data specs of the :class:`~GraphTensor.Spec` instance.

            :meth:`~data_spec` corresponds to :meth:`~GraphTensor.data`.

            When working with specs returned from `data_spec`, make sure to 
            also work with values returned from `data` of the associated 
            :class:`~GraphTensor` instance. If not, conflicts will likely occur.

            This property is implemented to be more selective and have more 
            control over what data specs should be unpacked from the 
            :class:`~GraphTensor.Spec` instance. 

            Unfortunately, `tf.nest` ops cannot be used directly on the 
            :class:`~GraphTensor.Spec` instance as it will expand all composites 
            including `tf.RaggedTensorSpec` specs. E.g., when performing
            `tf.nest.map_structure` it will flatten `tf.RaggedTensorSpec` specs 
            into its composites, which is an undesired behavior. By returning a dict 
            of nested data, `tf.nest` ops can be used on the dict without specifying
            expand_composites=True, resulting in `tf.RaggedTensor.Spec` specs not being 
            flattened. Furthermore, it allows us to unpack the `auxiliary` data
            specs approprietly. 
            
            Returns:
                A dictionary with nested data specs.
            '''

            return _get_data_or_data_spec(self)

        def is_scalar(self) -> None:

            '''Checks whether the :class:`~GraphTensor.Spec` instance is a "scalar".

            A "scalar" is loosely defined, but basically means that the 
            :class:`~GraphTensor.Spec` instance is "unbatched". The method should
            rarely be used. Currently only used for the custom batch encoder 
            to flag that the associated :class:`~GraphTensor` instance should not be 
            separated when encoded (in the custom batch encoder).
            
            Returns:
                A boolean indicating whether the :class:`~GraphTensor.Spec` instance
                is a "scalar".
            '''

            return self.sizes.shape.rank == 0
        
        def is_ragged(self) -> bool:

            '''Checks whether the :class:`~GraphTensor.Spec` instance is in its ragged state.
            
            Compared to :meth:`~GraphTensor.is_ragged`, this method also needs
            to check the ragged rank. Perhaps unintuitively, the result of this
            method will be False if ragged rank is 0. I.e., the spec is 
            considered non-ragged if encoding nested `tf.RaggedTensorSpec` specs
            of ragged rank 0.

            Returns:
                A boolean indicating whether the :class:`~GraphTensor.Spec` instance
                is in its ragged state.
            '''

            return (
                isinstance(self.node_feature, tf.RaggedTensorSpec) 
                and self.node_feature.ragged_rank == 1)

        def with_shape(self, shape):
            # Keras API
            return self.__class__(
                sizes=self.sizes,
                node_feature=self.node_feature,
                edge_src=self.edge_src,
                edge_dst=self.edge_dst,
                edge_feature=self.edge_feature,
                edge_weight=self.edge_weight,
                node_position=self.node_position,
                **self.auxiliary)
        

class _GraphTensorIterator:

    __slots__ = ('_iterable', '_index', '_limit')

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


# Helper functions for the GraphTensor:

def _get_data_or_data_spec(
    gt: GraphTensor
) -> typing.Mapping[str, TensorOrTensorSpec]:
    data_or_data_spec = {}
    for key in DEFAULT_FIELDS:
        if key == 'sizes':
            continue
        val = gt.__dict__[key]
        if val is not None:
            data_or_data_spec[key] = val
    for key, val in gt.auxiliary.items():
        data_or_data_spec[key] = val
    return data_or_data_spec

def _sizes_from_graph_indicator(
    graph_indicator: tf.Tensor, 
    length: typing.Optional[int] = None
) -> tf.Tensor:
    orig_dtype = graph_indicator.dtype
    graph_indicator = tf.cast(graph_indicator, tf.int32)
    return tf.math.bincount(graph_indicator, dtype=orig_dtype, minlength=length)

def _get_spec(x: TensorSpecOrShape, dtype, flat_ndims, force_shape=None):

    # TODO: Improve this

    if isinstance(x, tf.TensorSpec):
        # Make sure first dimension is None to indicate variable size.
        return tf.TensorSpec(shape=[None] + x.shape[1:], dtype=x.dtype)
    elif isinstance(x, tf.RaggedTensorSpec):
        return x 
    elif x is None:
        if force_shape is None:
            return x 
        x = force_shape

    x = tf.TensorShape(x)
    if x.rank > flat_ndims:
        # As rank is greater than flat_ndims it is ragged
        return tf.RaggedTensorSpec(x, dtype, 1, tf.int64)
    return tf.TensorSpec(x, dtype)

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

    if not tf.is_tensor(limit):
        limit = tf.convert_to_tensor(limit, dtype=DEFAULT_INDEX_DTYPE)

    if stop is None:
        stop = limit
    else:
        stop = tf.cast(stop, dtype=limit.dtype)
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
        start = tf.constant(0, dtype=limit.dtype)
    else:
        start = tf.cast(start, dtype=limit.dtype)
        start = tf.cond(
            start < 0,
            lambda: tf.maximum(limit + start, 0),
            lambda: start
        )

    if step is None:
        step = tf.constant(1, dtype=limit.dtype)
    else:
        step = tf.cast(step, dtype=limit.dtype)

    start = tf.cond(start > stop, lambda: stop, lambda: start)

    return tf.range(start, stop, step)

def _convert_to_ragged_tensor(inputs, dtype=tf.int64):
    row_lengths = [tf.shape(x, out_type=dtype)[0] for x in inputs]
    try:
        inputs_concatenated = tf.concat(inputs, axis=0)
    except tf.errors.InvalidArgumentError:
        inputs_concatenated = tf.concat([
            x for (x, rl) in zip(inputs, row_lengths) if rl != 0], axis=0)
    return tf.RaggedTensor.from_row_lengths(
        inputs_concatenated, tf.cast(row_lengths, dtype))

def _has_ragged_second_dimension(arr):
    
    if not isinstance(arr, Iterable):
        return False
    
    lengths = []
    for i, subarr in enumerate(arr):
        if np.isscalar(subarr):
            lengths.append(-1)
        else:
            lengths.append(len(subarr))
        
        if lengths[0] != lengths[i]:
            return True
    
    return False

def _maybe_convert_to_tensor(inputs):
    if inputs is None or tf.is_tensor(inputs):
        return inputs
    if _has_ragged_second_dimension(inputs):
        if not tf.executing_eagerly():
            raise ValueError(
                'Make sure to pass tf.RaggedTensor values when instantiating '
                'a ragged GraphTensor in graph mode (non-eagerly).')
        return _convert_to_ragged_tensor(inputs) 
    return tf.convert_to_tensor(inputs)



# Define assert statement and checks for GraphTensor:


def _assert(test: typing.Union[bool, tf.Tensor], message: str) -> None:
    'Helper function to make TF assert statements.'
    assert_op = tf.Assert(tf.reduce_all(test), [message])
    if hasattr(assert_op, 'mark_used'):
        assert_op.mark_used()

def _assert_compatible_outer_shape(
    field: str, 
    old_val: Tensor, 
    new_val: Tensor
) -> None:

    assert_message = (
        'Found incompatible outer shapes. Specifically updated {0} has outer '
        'shape {1}, but was expected to have outer shape {2}.')
    
    if isinstance(old_val, tf.Tensor):
        old_shape, new_shape = tf.shape(old_val)[:1], tf.shape(new_val)[:1]
        test = (old_shape == new_shape)
        _assert(test, assert_message.format(field, new_shape, old_shape))
    else:
        old_shape, new_shape = tf.shape(old_val)[:2], tf.shape(new_val)[:2]
        test = (old_shape == new_shape)
        _assert(test, assert_message.format(field, new_shape, old_shape))

def _assert_no_intersubgraph_edges(gt: GraphTensor) -> None:

    assert_message = (
        'Found edges which connects different subgraphs. '
        'Specifically, at least one source node and associated destination '
        'node are contained in different subgraphs defined by the `sizes` '
        'data field. Currently, this is not allowed as it cannot be determined '
        'which row they should be partioned in.')
    
    subgraphs_src = tf.gather(gt.graph_indicator, gt.edge_src)
    subgraphs_dst = tf.gather(gt.graph_indicator, gt.edge_dst)
    test = tf.reduce_sum(subgraphs_dst - subgraphs_src) == 0
    _assert(test, assert_message)

def _check_edges(gt: GraphTensor) -> None:

    error_message = (
        'Found at least one edge that does not correspond to any node. '
        'Specifically, found entry {0} of {1}, which is greater than or equal '
        'to the number of nodes of the (sub)graph, which is {2}.')
    
    if gt.is_ragged():
        num_nodes = tf.cast(gt.node_feature.row_lengths(), gt.edge_src.dtype)
        src_indices = tf.reduce_max(gt.edge_src, axis=1)
        dst_indices = tf.reduce_max(gt.edge_dst, axis=1)
        for (a, b) in zip(num_nodes, src_indices):
            if a <= b:
                raise ValueError(error_message.format(
                    b, 'edge_src', a))
        for (a, b) in zip(num_nodes, dst_indices):
            if a <= b:
                raise ValueError(error_message.format(
                    b, 'edge_dst', a))
    else:
        num_nodes = tf.shape(gt.node_feature, out_type=gt.edge_src.dtype)[0]
        max_src_index = tf.reduce_max(gt.edge_src)
        max_dst_index = tf.reduce_max(gt.edge_dst)
        if not num_nodes > max_src_index:
            raise ValueError(error_message.format(
                max_src_index, 'edge_src', num_nodes))
        if not num_nodes > max_dst_index:
            raise ValueError(error_message.format(
                max_dst_index, 'edge_src', num_nodes))

def _check_ranks(gt: GraphTensor):

    error_message = (
        'Found rank of {0} to be {1}. Maximum rank for {0} is {2}. ')
    
    is_ragged = gt.is_ragged()

    expected_maximum_rank = {
        'sizes':    1,
        'edge_src': 2 if is_ragged else 1,
        'edge_dst': 2 if is_ragged else 1,
    }
    for key, value in gt.data.items():
        if key in expected_maximum_rank:
            rank = value.shape.rank
            expected_rank = expected_maximum_rank[key]
            if rank != expected_rank:
                raise ValueError(
                    error_message.format(key, rank, expected_rank))

def _check_compatible_types(gt: GraphTensor) -> None:
    # Should be called before _assert_compatible_sizes
    error_message = (
        'Found nested tensors with different types. All nested tensors need '
        'to have the same types: either all tf.Tensor types or all '
        'tf.RaggedTensor types. Specifically, found {}')
    tensor_types = [
        isinstance(v, tf.Tensor) for (k, v) in gt.data.items() 
        if not k.startswith('_')]
    if not all(tensor_types) and any(tensor_types):
        raise TypeError(error_message.format(list(gt.data.values())))

def _check_compatible_sizes(gt: GraphTensor) -> None:

    error_message = (
        'Found incompatible sizes. Namely, data fields associated with nodes '
        'or edges did not have the expected number of nodes or edges '
        'respectively. If this was intended, make sure to prepend an '
        'underscore ("_") to the name of the data field to indicate '
        'it is static. Specifically, outer dimension of {0} is {1}, which is '
        'incompatible with shape of {2} which is {3}')

    ndims = 2 if gt.is_ragged() else 1

    def get_shape(x):
        return tf.shape(x)[:ndims]

    node_shape = get_shape(gt.node_feature)
    edge_shape = get_shape(gt.edge_src)

    for key, value in gt.data.items():
        if key in ['node_feature', 'edge_src']:
            continue
        if key in DEFAULT_NODE_FIELDS or key.startswith('node'):
            value_shape = get_shape(value)
            if not (value_shape == node_shape):
                raise ValueError(error_message.format(
                    'node_feature', node_shape, key, value_shape))
        elif key in DEFAULT_EDGE_FIELDS or key.startswith('edge'):
            value_shape = get_shape(value)
            if not (value_shape == edge_shape):
                raise ValueError(error_message.format(
                    'edge_src', edge_shape, key, value_shape))


# Override default tensorflow implemenations for the GraphTensor:


@dispatch_for_api(tf.shape, {'input': GraphTensor})
def graph_tensor_shape(
    input, 
    out_type=tf.int32, 
    name=None
):
    '''For now, tf.shape is implemented for the GraphTensor.

    As tf.keras.Model.predict performs tf.shape(...) on ExtensionTypes,
    tf.shape needed to be implemented for the GraphTensor.

    Note that tf.shape returns the flat shape of the GraphTensor. So
    instead of (num_subgraphs, num_nodes_per_subgraph[variable], num_features), 
    which is the result of .shape, tf.shape returns (num_nodes, num_features).
    '''
    node_feature = input.node_feature
    if isinstance(node_feature, tf.RaggedTensor):
        node_feature = node_feature.flat_values
    return tf.shape(input=node_feature, out_type=out_type, name=name)

@dispatch_for_api(tf.gather, {'params': GraphTensor})
def graph_tensor_gather(
    params,
    indices,
    validate_indices=None,
    axis=None,
    batch_dims=0,
    name=None
):
    
    'Gathers certain subgraphs from the GraphTensor instance.'

    if axis is not None and axis != 0:
        raise ValueError(
            'tf.gather on a GraphTensor instance requires axis to be None or 0.')

    graph_tensor = params
    gather_args = (indices, validate_indices, axis, batch_dims, name)

    is_ragged = graph_tensor.is_ragged()

    if not is_ragged:
        graph_tensor = graph_tensor.separate()

    data = tf.nest.map_structure(
        lambda x: tf.gather(x, *gather_args), graph_tensor.data)

    sizes = tf.gather(graph_tensor.sizes, *gather_args)
    graph_tensor = GraphTensor(sizes=sizes, **data)

    if not is_ragged and graph_tensor.is_ragged():
        return graph_tensor.merge()

    return graph_tensor

@dispatch_for_api(tf.concat, {'values': typing.List[GraphTensor]})
def graph_tensor_concat(
    values,
    axis=0,
    name='concat'
):
    'Concatenates a list of GraphTensor instances.'

    # TODO: Can be improved? 

    if axis is not None and axis != 0:
        raise ValueError(
            'tf.concat on GraphTensor instances require axis to be None or 0.')

    structure = values[0].data
    dtype = values[0].edge_src.dtype

    is_ragged = tf.nest.map_structure(GraphTensor.is_ragged, values)

    if 0 < sum(is_ragged) < len(is_ragged):
        raise ValueError(
            'Nested data of the GraphTensor instances do not have consistent '
            'types: found both tf.RaggedTensor values and tf.Tensor values.')
   
    is_ragged = is_ragged[0]

    sizes = [x.sizes for x in values]
    if sizes[0].shape.rank == 0:
        sizes = tf.stack(sizes, axis=0)
    else:
        sizes = tf.concat(sizes, axis=0)

    # Unpacks list of graph tensors:
    # [t_i, ...] -> [[c1_i, c2_i, ...], ...],
    # where "t" and "c" denote graph tensor and nested components respectively.
    flat_sequences = tf.nest.map_structure(
        lambda x: tf.nest.flatten(x.data, expand_composites=True), values)
    
    if is_ragged:
        # if nested ragged tensors:
        # [t_i, ...] -> [[c1a_i, c1b_i, c2a_i, c2b_i, ...], ...],
        # where "a" and "b" denote values and row splits respectively.
        # start from 1 as first is "sizes".
        row_splits_list = [flat_seq[1::2] for flat_seq in flat_sequences]
        flat_sequences = [flat_seq[0::2] for flat_seq in flat_sequences]
        flat_sequences = tf.nest.map_structure(
            lambda values, row_splits: tf.RaggedTensor.from_row_splits(
                values, tf.cast(row_splits, sizes.dtype)),
            flat_sequences, row_splits_list)
    
    # Groups data fields if the graph tensors together:
    # [[c1_i, c2_i, ...], ...] -> [[c1_i, c1_i+1, ...], [c2_i, c2_i+1, ...], ...] 
    flat_sequences = list(zip(*flat_sequences))

    if is_ragged:
        flat_sequence_stacked = [
            tf.concat(x, axis=0) for x in flat_sequences]
    else:
        flat_sequence_stacked = [
            _convert_to_ragged_tensor(x, dtype) for x in flat_sequences]

    values = tf.nest.pack_sequence_as(structure, flat_sequence_stacked)
    
    values['sizes'] = sizes

    values = GraphTensor(**values)

    if is_ragged:
        return values.merge().separate()

    values = values.merge()

    return values

@dispatch_for_api(tf.split, {'value': GraphTensor})
def graph_tensor_split(
    value, 
    num_or_size_splits, 
    axis=0, 
    num=None, 
    name='split'
):
    
    if axis is not None and axis != 0:
        raise ValueError(
            'tf.split on GraphTensor instances require axis to be None or 0.')
    
    is_ragged = value.is_ragged()
    if not is_ragged:
        value = value.separate()
    
    data = value.data 

    data['sizes'] = value.sizes 

    data = tf.nest.map_structure(
        lambda x: tf.split(x, num_or_size_splits, axis, num, name),
        data
    )

    keys = data.keys()
    data = list(zip(*data.values()))

    if not is_ragged:
        return [GraphTensor(**dict(zip(keys, d))).merge() for d in data]
    return [GraphTensor(**dict(zip(keys, d))) for d in data]

@dispatch_for_api(tf.matmul, {
    'a': typing.Union[GraphTensor, tf.Tensor], 
    'b': typing.Union[GraphTensor, tf.Tensor]})
def graph_tensor_matmul(
    a,
    b,
    transpose_a=False,
    transpose_b=False,
    adjoint_a=False,
    adjoint_b=False,
    a_is_sparse=False,
    b_is_sparse=False,
    output_type=None
):
    '''Allows graph tensors to be matrix multiplied.

    Specifically, the `node_feature` field will be used for
    the matrix multiplication. This implementation makes it
    possible to pass graph tensors to `keras.layers.Dense`.
    '''
    if isinstance(a, GraphTensor):
        if a.is_ragged():
            a = a.merge()
        a = a.node_feature
    if isinstance(b, GraphTensor):
        if b.is_ragged():
            b = b.merge()
        b = b.node_feature
    return tf.matmul(
        a=a, 
        b=b,
        transpose_a=transpose_a, 
        transpose_b=transpose_b, 
        adjoint_a=adjoint_a,
        adjoint_b=adjoint_b, 
        a_is_sparse=a_is_sparse, 
        b_is_sparse=b_is_sparse, 
        output_type=output_type)

@dispatch_for_api(tf.boolean_mask, {'tensor': GraphTensor})
def graph_tensor_boolean_mask(tensor, mask, axis=None):
    
    '''Allows GraphTensor to be masked, via tf.boolean_mask.

    Conveniently, either subgraphs, nodes or edges can be masked out. 
    Masking here could be thought of as dropout, as the subgraphs, nodes
    or edges are completely dropped from the :class:`~GraphTensor` instance.

    Args:
        tensor (GraphTensor):
            A :class:`~GraphTensor` instance.
        mask (tf.Tensor):
            A 1-D tf.Tensor specifying the subgraph, node or edge mask. Size 
            should correspond to the number of subgraphs, nodes or edges 
            respectively.
        axis (int, str, None):
            If axis is set to None, 0, or 'graph', subgraphs will be masked;
            if axis is 'node' or 1, nodes will be masked; if axis is 'edge'
            or 2, edges will be masked. `axis` usually does not accept
            strings; however, as (1) the axis to perform subgraph, node or edge
            masking is always 0 anyways, and (2) additional arguments 
            could not be added, it was decided to use the `axis` argument
            to indicate whether nodes or edges should be masked.

    Returns:
        GraphTensor: Masked instance of a GraphTensor.

    '''

    if axis is None or axis == 0 or axis == 'graph':
        return _mask_subgraphs(tensor, mask)

    is_ragged = tensor.is_ragged()

    if is_ragged:
        tensor = tensor.merge()

    if isinstance(mask, tf.RaggedTensor):
        mask = mask.flat_values

    if axis == 1 or axis == 'node':
        tensor = _mask_nodes(tensor, mask)
    else:
        tensor = _mask_edges(tensor, mask)

    if is_ragged:
        tensor = tensor.separate()

    return tensor 

@dispatch_for_unary_elementwise_apis(GraphTensor)
def graph_tensor_unary_elementwise_op_handler(api_func, x):
    '''Allows all unary elementwise operations (such as `tf.math.abs`)
    to handle graph tensors.
    '''
    return x.update({'node_feature': api_func(x.node_feature)})

@dispatch_for_binary_elementwise_apis(
    typing.Union[GraphTensor, tf.Tensor, float], 
    typing.Union[GraphTensor, tf.Tensor, float]
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

def _mask_subgraphs(
    tensor: GraphTensor,
    subgraph_mask: tf.Tensor,
) -> GraphTensor:
    indices = tf.where(subgraph_mask)[:, 0]
    return tf.gather(tensor, indices)

def _mask_nodes(
    tensor: GraphTensor, 
    node_mask: tf.Tensor
) -> GraphTensor:
    
    num_nodes = tf.shape(tensor.node_feature, out_type=tensor.edge_src.dtype)[0]

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
    data = tensor.data

    graph_indicator = tensor.graph_indicator
    graph_indicator = tf.boolean_mask(graph_indicator, node_mask)

    # Add new (masked) edge_dst and edge_src
    data['edge_src'] = edge_src
    data['edge_dst'] = edge_dst
    
    # Apply masks on the rest of the data and add to data dict
    # Both data associated with edges and nodes will be masked
    for key in data.keys():
        if key not in ['edge_src', 'edge_dst']: # if not yet masked
            if key in DEFAULT_NODE_FIELDS or key.startswith('node'):
                data[key] = tf.boolean_mask(data[key], node_mask)
            elif key in DEFAULT_EDGE_FIELDS or key.startswith('edge'):
                data[key] = tf.boolean_mask(data[key], edge_mask)
            elif not key.startswith('_'):
                raise ValueError(
                    f'Data field {key} not recognized. For user specified data '
                     'fields, either prepend `node` or `edge` to the name, '
                     'or `_` to indicate a static data field which will just '
                     'be passed along with the GraphTensor instance as is.')
    
    sizes = _sizes_from_graph_indicator(
        graph_indicator, tensor.num_subgraphs)
    
    return GraphTensor(sizes=sizes, **data)

def _mask_edges(
    tensor: GraphTensor, 
    edge_mask: tf.Tensor
) -> GraphTensor:
    # Obtain data of the GraphTensor
    data = tensor.data
    # Mask all data associated with edges
    for key in data.keys():
        if key in DEFAULT_EDGE_FIELDS or key.startswith('edge'):
             data[key] = tf.boolean_mask(data[key], edge_mask)
    return GraphTensor(**data)