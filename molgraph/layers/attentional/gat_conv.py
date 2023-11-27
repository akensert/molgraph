import tensorflow as tf
from tensorflow import keras

from keras import initializers
from keras import regularizers
from keras import constraints
from keras import activations

from typing import Optional
from typing import Callable
from typing import Union

from molgraph.internal import register_keras_serializable 

from molgraph.tensors.graph_tensor import GraphTensor

from molgraph.layers import gnn_layer
from molgraph.layers import gnn_ops 


@register_keras_serializable(package='molgraph')
class GATConv(gnn_layer.GNNLayer):

    '''Multi-head graph attention layer (GAT).

    The implementation is based on Velickovic et al. (2018) [#]_ and
    Dwivedi et al. (2022) [#]_.

    Example usage:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     sizes=[2, 3],
    ...     node_feature=[[1., 0.], [1., 0.], [1., 0.], [1., 0.], [0., 1.]],
    ...     edge_src=[1, 0, 3, 4, 2, 4, 3, 2],
    ...     edge_dst=[0, 1, 2, 2, 3, 3, 4, 4],
    ... )
    >>> gnn_model = tf.keras.Sequential([
    ...     molgraph.layers.GATConv(units=16),
    ...     molgraph.layers.GATConv(units=16),
    ...     molgraph.layers.GATConv(units=16),
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
    ...     molgraph.layers.GATConv(units=16, use_edge_features=True),
    ...     molgraph.layers.GATConv(units=16, use_edge_features=True),
    ...     molgraph.layers.GATConv(units=16, use_edge_features=True),
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
        normalization: (None, str, bool):
            Whether to apply layer normalization to the output. If batch 
            normalization is desired, pass 'batch_norm'. Default to None.
        residual: (bool)
            Whether to add skip connection to the output. Default to True.
        dropout: (float, None):
            Dropout applied to the output of the layer. Default to None.
        attention_activation (tf.keras.activations.Activation, callable, str, None):
            Activation function applied to the the attention scores.
            Default to 'leaky_relu'.
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

    References:
        .. [#] https://arxiv.org/pdf/1710.10903.pdf
        .. [#] https://arxiv.org/pdf/2003.00982.pdf

    '''

    def __init__(
        self,
        units: Optional[int] = 128,
        num_heads: int = 8,
        merge_mode: Optional[str] = 'concat',
        self_projection: bool = True,
        normalization: Union[None, str, bool] = None,
        residual: bool = True,
        dropout: Optional[float] = None,
        attention_activation: Union[
            None, str, Callable[[tf.Tensor], tf.Tensor]] = 'leaky_relu',
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
        super().__init__(
            units=units,
            normalization=normalization,
            residual=residual,
            dropout=dropout,
            activation=activation,
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
        self.activation = activations.get('elu')
        self.attention_activation = activations.get(attention_activation)

    def _build(self, graph_tensor_spec: GraphTensor.Spec) -> None:

        if self.merge_mode == 'concat':
            if not self.units or (self.units % self.num_heads != 0):
                raise ValueError(
                    '`merge_mode` was set to `concat` and hence ' +
                    ' need `units` to be divisble by `num_heads`')
            self.units //= self.num_heads
            
        if self.use_edge_features:

            self.edge_projection = self.get_einsum_dense(
                'ij,jkh->ihk', (self.num_heads, self.units))

            if self.update_edge_features:
                self.edge_out_projection = self.get_einsum_dense(
                    'ihj,jkh->ihk', (self.num_heads, self.units))

        self.node_projection = self.get_einsum_dense(
            'ij,jkh->ihk', (self.num_heads, self.units))

        if self.apply_self_projection:
            self.self_projection = self.get_einsum_dense(
                'ij,jkh->ihk', (self.num_heads, self.units))

        self.attention_projection = self.get_einsum_dense(
            'ihj,jhk->ihk', (self.num_heads, 1))

        if self.merge_mode == 'concat':
            self.units *= self.num_heads

    def _call(self, tensor: GraphTensor) -> GraphTensor:

        if self.apply_self_projection:
            node_feature_residual = self.self_projection(tensor.node_feature)
        else:
            node_feature_residual = None

        # Edge dependent (i.e., `tensor.edge_src is not None`), from here
        node_feature = self.node_projection(tensor.node_feature)

        attention_feature = tf.concat([
            tf.gather(node_feature, tensor.edge_dst),
            tf.gather(node_feature, tensor.edge_src)], axis=-1)

        if self.use_edge_features:
            edge_feature = self.edge_projection(tensor.edge_feature)
            attention_feature = tf.concat([
                attention_feature, edge_feature], axis=-1)

            if self.update_edge_features:
                edge_feature = self.edge_out_projection(attention_feature)
                edge_feature = gnn_ops.reduce_features(
                    feature=edge_feature, 
                    mode=self.merge_mode,
                    output_units=self.units)
                tensor = tensor.update({'edge_feature': edge_feature})

        edge_weights = self.attention_projection(attention_feature)
        edge_weights = self.attention_activation(edge_weights)

        tensor = tensor.update({
            'node_feature': node_feature, 'edge_weight': edge_weights})
        
        return tensor.propagate(
            activation=self.activation,
            normalize=True, 
            reduction=self.merge_mode,
            residual=node_feature_residual)

    def get_config(self):
        base_config = super().get_config()
        config = {
            'num_heads': self.num_heads,
            'merge_mode': self.merge_mode,
            'self_projection': self.apply_self_projection,
            'attention_activation': activations.serialize(self.attention_activation),
        }
        base_config.update(config)
        return base_config
