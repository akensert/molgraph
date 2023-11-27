import tensorflow as tf
from tensorflow import keras

from keras import layers
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import activations

import abc

from warnings import warn

from typing import Optional
from typing import Callable
from typing import Union
from typing import List
from typing import Type

from molgraph.internal import register_keras_serializable 

from molgraph.tensors.graph_tensor import GraphTensor


@register_keras_serializable(package='molgraph')
class GNNLayer(layers.Layer, metaclass=abc.ABCMeta):

    '''Base layer for the built-in GNN layers. 
    
    Can also be used to create new GNN layers.
    
    Example usage:

    >>> class MyGCNLayer(molgraph.layers.GNNLayer):
    ...
    ...     def __init__(self, units, **kwargs):
    ...         super().__init__(
    ...             units=units, 
    ...             **kwargs)
    ...
    ...     def _call(self, graph_tensor):
    ...         node_feature_transformed = self.projection(
    ...             graph_tensor.node_feature)
    ...         graph_tensor = graph_tensor.update({
    ...             'node_feature': node_feature_transformed})
    ...         return graph_tensor.propagate()
    ...
    ...     def _build(self, graph_tensor_spec):
    ...         self.projection = self.get_dense(self.units)
    ...
    >>> my_gcn_layer = MyGCNLayer(32)
    >>> my_gcn_layer.compute_output_shape(tf.TensorShape((None, 8)))
    TensorShape([None, 32])

    Args:
        units (int, None):
            Number of output units.
        normalization: (None, str, bool):
            Whether to apply layer normalization to the output. If batch 
            normalization is desired, pass 'batch_norm'. Default to None.
        residual: (bool)
            Whether to add skip connection to the output. Default to True.
        dropout: (float, None):
            Dropout applied to the output of the layer. Default to None.
        activation (tf.keras.activations.Activation, callable, str, None):
            Activation function applied to the output of the layer.
            Default to 'relu'.
        use_bias (bool):
            Whether the layer should use biases. Default to True.
        kernel_initializer (tf.keras.initializers.Initializer, str):
            Initializer function for the kernels. Default to
            tf.keras.initializers.TruncatedNormal(stddev=0.005).
        bias_initializer (tf.keras.initializers.Initializer, str):
            Initializer function for the biases. Default to
            tf.keras.initializers.Constant(0.).
        kernel_regularizer (tf.keras.regularizers.Regularizer, None):
            Regularizer function applied to the kernels. Default to None.
        bias_regularizer (tf.keras.regularizers.Regularizer, None):
            Regularizer function applied to the biases. Default to None.
        activity_regularizer (tf.keras.regularizers.Regularizer, None):
            Regularizer function applied to the final output of the layer.
            Default to None.
        kernel_constraint (tf.keras.constraints.Constraint, None):
            Constraint function applied to the kernels. Default to None.
        bias_constraint (tf.keras.constraints.Constraint, None):
            Constraint function applied to the biases. Default to None.
        **kwargs: Valid (optional) keyword arguments are:

            *   `name` (str): Name of the layer instance.
            *   `update_step` (tf.keras.layers.Layer): Applies post-processing 
                step on the output (produced by `_call`). If passed, 
                `normalization`, `residual`, `activation` and `dropout` 
                parameters will be ignored. If None, a default post-processing 
                step will be used (taking into consideration the aforementioned 
                parameters). Default to None.
            *   `use_edge_features`: Whether or not to use edge features. 
                Only relevant if edge features exist. If None, and edge 
                features exist, it will be set to True. Default to None.
            *   `update_edge_features` (bool): Specifies whether edge features 
                should be updated along with node features, including the 
                post-processing step. Only relevant if edge features exist. 
                It is important that GNN layers which updates its edge features
                for the next layer sets this to True. Default to False. 
    '''

    def __init__(
        self,
        units: int,
        normalization: Union[None, str, bool] = None,
        residual: Optional[bool] = False,
        dropout: Optional[float] = None,
        activation: Union[Callable[[tf.Tensor], tf.Tensor], str, None] = None,
        use_bias: bool = False,
        kernel_initializer: Union[str, initializers.Initializer, None] = None,
        bias_initializer: Union[str, initializers.Initializer, None] = None,
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_regularizer: Optional[regularizers.Regularizer] = None,
        activity_regularizer: Optional[regularizers.Regularizer] = None,
        kernel_constraint: Optional[constraints.Constraint] = None,
        bias_constraint: Optional[constraints.Constraint] = None,
        **kwargs
    ) -> None:

        self.update_step_fn = kwargs.pop('update_step', None)
        self.use_edge_features = kwargs.pop('use_edge_features', None) 
        self.update_edge_features = kwargs.pop('update_edge_features', False)

        layers.Layer.__init__(self, **kwargs)

        # get_kernel, get_bias, get_dense, get_einsum_dense parameters:
        self.units = units
        self._use_bias = use_bias
        if kernel_initializer is None:
            kernel_initializer = initializers.TruncatedNormal(stddev=0.005)
        self._kernel_initializer = initializers.get(kernel_initializer)
        if bias_initializer is None:
            bias_initializer = initializers.Constant(0.)
        self._bias_initializer = initializers.get(bias_initializer)
        self._kernel_regularizer = regularizers.get(kernel_regularizer)
        self._bias_regularizer = regularizers.get(bias_regularizer)
        self._activity_regularizer = regularizers.get(activity_regularizer)
        self._kernel_constraint = constraints.get(kernel_constraint)
        self._bias_constraint = constraints.get(bias_constraint)

        # _DefaultUpdateStep parameters (ignored if update_step is supplied):
        self._normalization = normalization
        self._activation = activations.get(activation)
        self._residual = residual
        self._dropout = dropout

        self._graph_tensor_spec = None
        self._built = False

    @abc.abstractmethod
    def _call(
        self, 
        graph_tensor: GraphTensor,
    ) -> GraphTensor:
        '''Calls the derived GNN layer. 

        Wrapped in :meth:`~call`.

        Args:
            graph_tensor (GraphTensor):
                Input to the layer.

        Returns:
            GraphTensor:
                A graph tensor with updated features. For some layers, only 
                the ``node_feature`` data is updated, for other layers both the 
                ``node_feature`` and the ``edge_feature`` data are updated.
        '''

    @abc.abstractmethod
    def _build(
        self,
        graph_tensor_spec: GraphTensor.Spec,
    ) -> None:
        '''Builds the derived GNN layer.
        
        Wrapped in :meth:`~build_from_signature`.

        Args:
            graph_tensor_spec (GraphTensor.Spec):
                The spec corresponding to a graph tensor instance.
        '''

    def call(
        self, 
        graph_tensor: GraphTensor, 
    ) -> GraphTensor:
        '''Wraps :meth:`~_call`, applying pre- and post-processing steps.

        Automatically builds the layer on first call.
        
        Args:
            graph_tensor (GraphTensor):
                An graph tensor instance.

        Returns:
            GraphTensor: An updated graph tensor instance. 
        '''
        
        graph_tensor_orig = graph_tensor
        if graph_tensor.is_ragged():
            graph_tensor = graph_tensor.merge()

        if not self._built:
            self.build_from_signature(graph_tensor)

        graph_tensor_updated = self._update(
            self._call(graph_tensor), graph_tensor)

        return graph_tensor_orig.update({
            k: v for (k, v) in graph_tensor_updated.data.items() if
            k not in ['edge_dst', 'edge_src', 'graph_indicator']
        })
    
    def build_from_signature(
        self,
        graph_tensor: GraphTensor,
    ) -> None:

        '''Builds the layer based on 
        a :class:`~molgraph.tensors.GraphTensor` or 
        a :class:`~molgraph.tensors.GraphTensor.Spec`.
        
        Automatically invoked on first :meth:`~call`.

        Args:
            graph_tensor (GraphTensor, GraphTensor.Spec):
                A graph tensor instance or a spec of a graph tensor instance.
        '''

        self._built = True

        update_step: layers.Layer = self.update_step_fn 

        self._graph_tensor_spec, (node_dim, edge_dim) = _get_spec_and_dims(
            graph_tensor)
        
        if edge_dim is None:
            self.use_edge_features = False
        elif self.use_edge_features is None:
            self.use_edge_features = True

        with tf.init_scope():

            self.node_dim = node_dim
        
            if not self.units:
                self.units = self.node_dim

            if self.units != self.node_dim and self._residual:
                self.node_feature_resample = self.get_dense(self.units)
            else:
                self.node_feature_resample = None

            self._node_update_step_fn = (
                _DefaultUpdateStep(
                    activation=self._activation, 
                    dropout=self._dropout, 
                    residual=self._residual, 
                    normalization=self._normalization
                ) if update_step is None else update_step
            )
   
            self.update_edge_features = (
                self.update_edge_features and self.use_edge_features)

            if self.update_edge_features:

                self.edge_dim = edge_dim

                if self.units != self.edge_dim and self._residual:
                    self.edge_feature_resample = self.get_dense(self.units)
                else:
                    self.edge_feature_resample = None

                self._edge_update_step_fn = (
                    _DefaultUpdateStep(
                        activation=self._activation, 
                        dropout=self._dropout, 
                        residual=self._residual, 
                        normalization=self._normalization
                    ) if update_step is None else 
                    update_step.from_config(update_step.get_config())
                )

            self._build(self._graph_tensor_spec)
    
    def get_kernel(
        self,
        shape: tf.TensorShape,
        dtype: tf.dtypes.DType = tf.float32,
        name: str = 'kernel'
    ) -> tf.Variable:
        '''Obtain (trainable) kernel weights based on parameters passed to 
        :meth:`~__init__`.

        Args:
            shape (tf.TensorShape):
                The shape of the weights.
            dtype (tf.dtypes.DType):
                The dtype of the weights. Default to `tf.float32`.
            name (str):
                The name of the weights. Default to 'kernel'.

        Returns:
            tf.Variable: Trainable bias weights.
        '''
        common_kwargs = self._get_common_kwargs()

        return self.add_weight(
            name=name,
            shape=shape,
            dtype=dtype,
            initializer=common_kwargs['kernel_initializer'],
            regularizer=common_kwargs['kernel_regularizer'],
            constraint=common_kwargs['kernel_constraint'],
            trainable=True
        )

    def get_bias(self,
        shape: tf.TensorShape,
        dtype: tf.dtypes.DType = tf.float32,
        name: str = 'bias'
    ) -> tf.Variable:
        '''Obtain (trainable) bias weights based on parameters passed to 
        :meth:`~__init__`.

        Args:
            shape (tf.TensorShape):
                The shape of the weights.
            dtype (tf.dtypes.DType):
                The dtype of the weights. Default to `tf.float32`.
            name (str):
                The name of the weights. Default to 'bias'.

        Returns:
            tf.Variable: Trainable bias weights.
        '''
        common_kwargs = self._get_common_kwargs()

        return self.add_weight(
            name=name,
            shape=shape,
            dtype=dtype,
            initializer=common_kwargs['bias_initializer'],
            regularizer=common_kwargs['bias_regularizer'],
            constraint=common_kwargs['bias_constraint'],
            trainable=True
        )

    def get_dense(
        self,
        units: int,
        activation: Union[Callable[[tf.Tensor], tf.Tensor], str, None] = None,
        use_bias: Optional[bool] = None,
    ) -> layers.Dense:
        '''Obtain a `Dense` layer based on parameters passed to 
        :meth:`~__init__`.

        Args:
            units (int):
                Number of units of the dense layer.
            activation (None, callable):
                The activation to be applied to the output of layer.
                Default to None.
            use_bias (None, bool):
                Whether bias should be used. If None, `use_bias` parameter
                passed to ``__init__`` will be used. Default to None.

        Returns:
            layers.Dense: The dense layer.
        '''
        if use_bias is None and self._use_bias:
            use_bias = self._use_bias

        return layers.Dense(
            units,
            activation=activation,
            use_bias=use_bias,
            **self._get_common_kwargs())

    def get_einsum_dense(
        self,
        equation: str,
        output_shape: tf.TensorShape,
        activation: Union[Callable[[tf.Tensor], tf.Tensor], str, None] = None,
        bias_axes: Optional[str] = None,
    ) -> layers.EinsumDense:
        '''Obtain an `EinsumDense` layer based on parameters passed to 
        :meth:`~__init__`.

        Args:
            equation (str):
                The einsum formula.
            output_shape (tf.TensorShape):
                The output shape (excluding batch dimension or any dimensions 
                represented by ellipses).
            activation (None, callable):
                The activation to be applied to the output of the layer. 
                Default to None.
            bias_axes (None, str):
                A string containing the output dimension(s) to apply a bias to.
                The string characters should correspond to `equation`. If None
                and ``__init__(..., use_bias=True)``, the bias axes will be 
                automatically derived. If None and 
                ``__init__(..., use_bias=False)``, no bias will be applied.

        Returns:
            layers.EinsumDense: The einsum dense layer.
        '''
        if bias_axes is None and self._use_bias:
            bias_axes = equation.split('->')[-1][1:]
            if len(bias_axes) == 0:
                bias_axes = None

        return layers.EinsumDense(
            equation,
            output_shape,
            activation=activation,
            bias_axes=bias_axes,
            **self._get_common_kwargs())

    def compute_output_shape(
        self,
        input_shape: tf.TensorShape
    ) -> tf.TensorShape:
        '''Computes the output shape of the layer based on an input shape.

        Args:
            input_shape (tf.TensorShape):
                The shape of a :class:`~molgraph.tensors.GraphTensor` instance.

        Returns:
            tf.TensorShape: The shape corresponding to the outputted (updated) 
            :class:`~molgraph.tensors.GraphTensor` instance.
        '''
        inner_dim = self.units
        if getattr(self, 'merge_mode', None) == 'concat':
            if self._built:
                if hasattr(self, 'num_heads'):
                    inner_dim *= self.num_heads
                elif hasattr(self, 'num_kernels'):
                    inner_dim *= getattr(self, 'num_kernels', 1)
        return tf.TensorShape(input_shape[:-1]).concatenate([inner_dim])

    def compute_output_signature(
        self,
        input_signature: GraphTensor.Spec,
    ) -> GraphTensor.Spec:
        '''Computes the output signature of the layer based on an input signature.
        
        Args:
            input_signature (GraphTensor.Spec):
                The spec of a :class:`~molgraph.tensors.GraphTensor` instance.
        
        Returns:
            GraphTensor.Spec: The spec corrsponding to the outputted (updated)
            :class:`~molgraph.tensors.GraphTensor` instance.
        '''
        def _update_spec(
            spec: Union[tf.TensorSpec, tf.RaggedTensorSpec],
        ) -> Union[tf.TensorSpec, tf.RaggedTensorSpec]:
            if isinstance(spec, tf.TensorSpec):
                return tf.TensorSpec(
                    shape=self.compute_output_shape(spec.shape),
                    dtype=spec.dtype)
            return tf.RaggedTensorSpec(
                shape=self.compute_output_shape(spec.shape),
                dtype=spec.dtype,
                ragged_rank=spec.ragged_rank,
                row_splits_dtype=spec.row_splits_dtype,
                flat_values_spec=spec.flat_values_spec)
    
        data_spec = input_signature.data_spec
        data_spec['node_feature'] = _update_spec(data_spec['node_feature'])
        if self.update_edge_features:
            data_spec['edge_feature'] = _update_spec(data_spec['edge_feature'])
        return input_signature.__class__(**data_spec)

    @classmethod
    def from_config(cls: Type['GNNLayer'], config: dict) -> 'GNNLayer':
        '''Initializes and builds the layer based on a configuration.
        
        Args:
            config (dict):
                A Python dictionary of parameters to initialize and build a 
                layer instance. The config is usually obtained from 
                ``get_config()`` of another layer instance (of the same class).
        
        Returns:
            A layer instance.
        '''
        graph_tensor_spec = config.pop('graph_tensor_spec', None)
        update_step = config.pop('update_step')
        config['update_step'] = (
            None if update_step is None else layers.deserialize(update_step))
        layer = cls(**config)
        if graph_tensor_spec is None:
            warn(
                (
                 'A GraphTensor.Spec could not be obtained from the config, '
                 'indicating that the layer from which the config was '
                 'previously obtained was not yet built. Proceeding to '
                 'initialize the layer without building it.'
                ),
                UserWarning,
                stacklevel=2
            )
        else:
            layer.build_from_signature(graph_tensor_spec)
        return layer

    def get_config(self) -> dict:
        '''Returns the configuration of the layer.
        
        Returns:
            Python dictionary with the layer's parameters. Can be used to
            initialize and build another layer instance (of the same class), 
            via ``from_config()``.
        '''
        config = super().get_config()
        config.update({
            'units': self.units,

            'use_edge_features': self.use_edge_features,
            'update_edge_features': self.update_edge_features,
            'update_step': layers.serialize(self.update_step_fn),
            
            'normalization':self._normalization,
            'activation': activations.serialize(self._activation),
            'residual': self._residual,
            'dropout': self._dropout,
            
            'use_bias': self._use_bias,
            'kernel_initializer':
                initializers.serialize(self._kernel_initializer),
            'bias_initializer':
                initializers.serialize(self._bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self._kernel_regularizer),
            'bias_regularizer':
                regularizers.serialize(self._bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self._activity_regularizer),
            'kernel_constraint':
                constraints.serialize(self._kernel_constraint),
            'bias_constraint':
                constraints.serialize(self._bias_constraint),
            'graph_tensor_spec':
                self._graph_tensor_spec,
        })
        return config

    def _update(
        self,
        tensor_input: GraphTensor,
        tensor_state: GraphTensor,
    ) -> GraphTensor:
        
        # Obtain features from previous (non-updated) graph tensor instance.
        node_feature_state = (
            tensor_state.node_feature if self.node_feature_resample is None 
            else self.node_feature_resample(tensor_state.node_feature))

        if self.update_edge_features:
            edge_feature_state = (
                tensor_state.edge_feature if self.edge_feature_resample is None 
                else self.edge_feature_resample(tensor_state.edge_feature))
            
        # Apply update step to updated (aggregated) graph tensor instance.
        data_updated = {}

        data_updated['node_feature'] = self._node_update_step_fn(
            tensor_input.node_feature, 
            node_feature_state)

        if self.update_edge_features:
            data_updated['edge_feature'] = self._edge_update_step_fn(
                tensor_input.edge_feature, 
                edge_feature_state)

        return tensor_input.update(data_updated)

    def _get_common_kwargs(self):
        common_kwargs = dict(
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
        )
        kernel_initializer = self._kernel_initializer.__class__.from_config(
            self._kernel_initializer.get_config()
        )
        bias_initializer = self._bias_initializer.__class__.from_config(
            self._bias_initializer.get_config()
        )
        common_kwargs["kernel_initializer"] = kernel_initializer
        common_kwargs["bias_initializer"] = bias_initializer
        return common_kwargs
    

class _DefaultUpdateStep(layers.Layer):

    def __init__(
        self, 
        normalization: Union[None, str, bool, layers.Layer] = None,
        activation: Union[Callable[[tf.Tensor], tf.Tensor], str, None] = 'relu',
        residual: Optional[bool] = True,
        dropout: Optional[float] = None,
        name: Optional[str] = 'DefaultUpdateStep',
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        if isinstance(normalization, str):
            self.normalization = (
                layers.BatchNormalization() 
                if normalization.lower().startswith('batch')
                else layers.LayerNormalization())
        elif isinstance(normalization, bool):
            if normalization:
                self.normalization = layers.LayerNormalization()
            else:
                self.normalization = None
        else:
            self.normalization = normalization

        self.activation = activations.get(activation)
        self.residual = residual
        self.dropout = layers.Dropout(dropout) if dropout is not None else None

    def call(
        self, 
        inputs: tf.Tensor,
        states: tf.Tensor,
    ) -> tf.Tensor:
        outputs = inputs
        if self.normalization is not None:
            outputs = self.normalization(outputs)
        if self.activation is not None:
            outputs = self.activation(outputs)
        if self.residual:
            outputs += states
        if self.dropout:
            outputs = self.dropout(outputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'normalization': layers.serialize(self.normalization),
            'activation': activations.serialize(self.activation),
            'dropout': layers.serialize(self.dropout),
            'residual': self.residual,
        })
        return config
    
    
def _get_spec_and_dims(
    graph_tensor: Union[GraphTensor, GraphTensor.Spec]
) -> GraphTensor.Spec:
    graph_tensor_spec = (
        graph_tensor if isinstance(graph_tensor, GraphTensor.Spec) else
        graph_tensor.spec)
    node_feature_spec = graph_tensor_spec.node_feature
    edge_feature_spec = graph_tensor_spec.edge_feature
    node_dim = (
        None if node_feature_spec is None else node_feature_spec.shape[-1])
    edge_dim = (
        None if edge_feature_spec is None else edge_feature_spec.shape[-1])
    return graph_tensor_spec, (node_dim, edge_dim)


class BaseLayer(GNNLayer):

    def __init__(self, *args, **kwargs):
        warn(
            (
                '`BaseLayer` will be depracated in the near future, please '
                'use `GNNLayer` instead.'
            ),
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)
