import tensorflow as tf
from tensorflow import keras
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import activations
from keras import layers

from typing import Optional
from typing import Callable
from typing import Union

from molgraph.internal import register_keras_serializable 

from molgraph.tensors.graph_tensor import GraphTensor

from molgraph.layers import gnn_layer
from molgraph.layers import gnn_ops


@register_keras_serializable(package='molgraph')
class GTConv(gnn_layer.GNNLayer):

    '''Graph transformer layer

    Implementation is based on Dwivedi et al. (2021) [#]_.

    Alias: ``GraphTransformerConv``

    Example usage:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     sizes=[2, 3],
    ...     node_feature=[[1., 0.], [1., 0.], [1., 0.], [1., 0.], [0., 1.]],
    ...     edge_src=[1, 0, 3, 4, 2, 4, 3, 2],
    ...     edge_dst=[0, 1, 2, 2, 3, 3, 4, 4],
    ... )
    >>> gnn_model = tf.keras.Sequential([
    ...     molgraph.layers.GTConv(units=16),
    ...     molgraph.layers.GTConv(units=16),
    ...     molgraph.layers.GTConv(units=16),
    ...     molgraph.layers.Readout(),
    ... ])
    >>> gnn_model(graph_tensor).shape
    TensorShape([2, 16])
    
    Including edge features:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     sizes=[2, 3],
    ...     node_feature=[[1., 0.], [1., 0.], [1., 0.], [1., 0.], [0., 1.]],
    ...     edge_feature=[[1., 0.], [0., 1.], [0., 1.], [0., 1.], 
    ...                   [1., 0.], [0., 1.], [1., 0.], [0., 1.]],
    ...     edge_src=[1, 0, 3, 4, 2, 4, 3, 2],
    ...     edge_dst=[0, 1, 2, 2, 3, 3, 4, 4],
    ... )
    >>> gnn_model = tf.keras.Sequential([
    ...     molgraph.layers.GTConv(units=16, use_edge_features=True),
    ...     molgraph.layers.GTConv(units=16, use_edge_features=True),
    ...     molgraph.layers.GTConv(units=16, use_edge_features=True),
    ... ])
    >>> output = gnn_model(graph_tensor)
    >>> output.node_feature.shape, output.edge_feature.shape
    (TensorShape([5, 16]), TensorShape([8, 16]))

    Args:
        units (int, None):
            Number of output units.
        num_heads (int):
            Number of attention heads. Default to 8.
        merge_mode (str):
            The strategy for merging the heads. Either of 'concat', 'sum',
            'mean' or None. If set to None, 'mean' is used. Default to 'concat'.
        self_projection (bool):
            Whether to apply self projection. Default to True.
        normalization (str, None):
            The type of normalization to use for the output. Either of
            'batch_norm', 'layer_norm' or None. Default to 'layer_norm'.
        residual: (bool)
            Whether to add skip connection to the output. Default to True.
        dropout: (float, None):
            Dropout applied to the output of the layer. Default to None.
        attention_activation (tf.keras.activations.Activation, callable, str, None):
            Activation function applied to the the attention scores. Default to None.
        activation (tf.keras.activations.Activation, callable, str, None):
            Activation function applied to the output of the layer. Default to 'relu'.
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

    References:
        .. [#] https://arxiv.org/pdf/2012.09699.pdf

    '''

    def __init__(
        self,
        units: Optional[int] = 128,
        num_heads: int = 8,
        merge_mode: Optional[str] = 'concat',
        self_projection: bool = True,
        normalization: Union[None, bool, str] = 'layer_norm',
        residual: bool = True,
        dropout: Optional[float] = None,
        activation: Union[None, str, Callable[[tf.Tensor], tf.Tensor]] = 'relu',
        use_bias: bool = True,
        kernel_initializer: Union[str, initializers.Initializer, None] = None,
        bias_initializer: Union[str, initializers.Initializer, None] = None,
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_regularizer: Optional[regularizers.Regularizer] = None,
        activity_regularizer: Optional[regularizers.Regularizer] = None,
        kernel_constraint: Optional[constraints.Constraint] = None,
        bias_constraint: Optional[constraints.Constraint] = None,
        **kwargs
    ):
        kwargs['update_edge_features'] = (
            kwargs.get('update_edge_features', True) and 
            kwargs.get('use_edge_features', True)
        )
        update_step = kwargs.pop(
            'update_step',
            _FeedForwardNetwork(
                units=units,
                normalization=normalization,
                activation=activation,
                residual=residual,
                dropout=dropout,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint,
            )
        )
        super().__init__(
            units=units,
            update_step=update_step,
            residual=residual,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

        self.num_heads = num_heads
        self.merge_mode = merge_mode
        self.apply_self_projection = self_projection

    def _build(self, graph_tensor_spec: GraphTensor.Spec) -> None:

        if self.merge_mode == 'concat':
            if not self.units or (self.units % self.num_heads != 0):
                raise ValueError(
                    "`merge_mode` was set to `concat` and hence " +
                    " need `units` to be divisble by `num_heads`")
            units_head = self.units // self.num_heads
        else:
            units_head = self.units

        self.query_projection = self.get_einsum_dense(
            'ij,jkh->ihk', (self.num_heads, units_head))

        self.key_projection = self.get_einsum_dense(
            'ij,jkh->ihk', (self.num_heads, units_head))

        self.value_projection = self.get_einsum_dense(
            'ij,jkh->ihk', (self.num_heads, units_head))

        if self.apply_self_projection:
            self.self_projection = self.get_einsum_dense(
                'ij,jkh->ihk', (self.num_heads, units_head))

        if self.use_edge_features:
            self.edge_gate_projection = self.get_einsum_dense(
                'ij,jkh->ihk', (self.num_heads, units_head))

    def _call(self, tensor: GraphTensor) -> GraphTensor:

        if self.apply_self_projection:
            node_feature_residual = self.self_projection(tensor.node_feature)
        else:
            node_feature_residual = None

        key = self.key_projection(tensor.node_feature)
        query = self.query_projection(tensor.node_feature)
        value = self.value_projection(tensor.node_feature)

        key = tf.gather(key, tensor.edge_src)
        query = tf.gather(query, tensor.edge_dst)

        attention_score = query * key

        attention_score = attention_score / tf.math.sqrt(float(self.units))

        if self.use_edge_features:
            edge_gate = self.edge_gate_projection(tensor.edge_feature)
            attention_score *= edge_gate
            if self.update_edge_features:
                edge_feature = gnn_ops.reduce_features(
                    feature=attention_score, 
                    mode=self.merge_mode, 
                    output_units=self.units)
                tensor = tensor.update({'edge_feature': edge_feature})

        tensor = tensor.update({
            'node_feature': value, 'edge_weight': attention_score})
        
        return tensor.propagate(
            normalize=True, 
            reduction=self.merge_mode,
            residual=node_feature_residual)
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            'num_heads': self.num_heads,
            'merge_mode': self.merge_mode,
            'self_projection': self.apply_self_projection,
        }
        base_config.update(config)
        return base_config


@register_keras_serializable(package='molgraph')
class _FeedForwardNetwork(layers.Layer):

    'Feed-forward network (FFN) of the graph transformer layer.'

    def __init__(
        self, 
        units: Optional[int] = None, 
        normalization: Union[None, str, bool] = 'layer_norm',
        activation: Union[Callable[[tf.Tensor], tf.Tensor], str, None] = None,
        residual: bool = True,
        dropout: Optional[float] = None,
        use_bias: bool = False,
        kernel_initializer: Union[str, initializers.Initializer, None] = None,
        bias_initializer: Union[str, initializers.Initializer, None] = None,
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_regularizer: Optional[regularizers.Regularizer] = None,
        activity_regularizer: Optional[regularizers.Regularizer] = None,
        kernel_constraint: Optional[constraints.Constraint] = None,
        bias_constraint: Optional[constraints.Constraint] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._units = units
        self._residual = residual
        self._normalization = normalization
        self._dropout = dropout
        self._activation = activations.get(activation)
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

        if self._dropout:
            self.dropout_1 = layers.Dropout(self._dropout)
            self.dropout_2 = layers.Dropout(self._dropout)

        if self._normalization:
            if str(self._normalization).startswith('batch'):
                self.normalization_1 = layers.BatchNormalization()
                self.normalization_2 = layers.BatchNormalization()
            else:
                self.normalization_1 = layers.LayerNormalization()
                self.normalization_2 = layers.LayerNormalization()
            
    def build(self, input_shape: tf.TensorShape) -> None:
        
        super().build(input_shape)

        if self._units is None:
            self._units = input_shape[-1]

        self.projection_1 = self._get_dense()
        self.projection_2 = self._get_dense()
        self.projection_3 = self._get_dense()

    def call(
        self, 
        inputs: tf.Tensor, 
        states: tf.Tensor
    ) -> tf.Tensor:
        
        x = inputs
        x_residual = states

        if self._dropout:
            x = self.dropout_1(x)

        x = self.projection_1(x)

        if self._residual:
            x += x_residual

        if self._normalization:
            x = self.normalization_1(x)

        x_residual = x

        x = self.projection_2(x)
        x = self._activation(x)

        if self._dropout:
            x = self.dropout_2(x)

        x = self.projection_3(x)

        if self._residual:
            x += x_residual

        if self._normalization:
            x = self.normalization_2(x)

        return x
    
    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'units': self._units,
            'normalization': self._normalization,
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
        })
        return config
    
    def _get_dense(self):
        common_kwargs = dict(
            units=self._units,
            activation=None,
            use_bias=self._use_bias,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint)
        kernel_initializer = self._kernel_initializer.__class__.from_config(
            self._kernel_initializer.get_config())
        bias_initializer = self._bias_initializer.__class__.from_config(
            self._bias_initializer.get_config())
        common_kwargs["kernel_initializer"] = kernel_initializer
        common_kwargs["bias_initializer"] = bias_initializer
        return layers.Dense(**common_kwargs)