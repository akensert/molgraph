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
class GINConv(gnn_layer.GNNLayer):

    '''Graph isomorphism convolution layer (GIN).

    Implementation based on Dwivedi et al. (2022) [#]_, Xu et al. (2019) [#]_, 
    and Hu et al. (2020) [#]_.

    Example usage:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     sizes=[2, 3],
    ...     node_feature=[[1., 0.], [1., 0.], [1., 0.], [1., 0.], [0., 1.]],
    ...     edge_src=[1, 0, 3, 4, 2, 4, 3, 2],
    ...     edge_dst=[0, 1, 2, 2, 3, 3, 4, 4],
    ... )
    >>> gnn_model = tf.keras.Sequential([
    ...     molgraph.layers.GINConv(units=16),
    ...     molgraph.layers.GINConv(units=16),
    ...     molgraph.layers.GINConv(units=16),
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
    ...     molgraph.layers.GINConv(units=16, use_edge_features=True),
    ...     molgraph.layers.GINConv(units=16, use_edge_features=True),
    ...     molgraph.layers.GINConv(units=16, use_edge_features=True),
    ... ])
    >>> output = gnn_model(graph_tensor)
    >>> output.node_feature.shape, output.edge_feature.shape
    (TensorShape([5, 16]), TensorShape([8, 16]))

    Args:
        units (int, None):
            Number of output units.
        apply_relu_activation (bool):
            Whether to apply relu activation before aggregation step. Only relevant
            if use_edge_features is set to True. Default to False.
        self_projection (bool):
            Whether to apply self projection. Default to True.
        normalization: (None, str, bool):
            Whether to apply layer normalization to the output. If batch 
            normalization is desired, pass 'batch_norm'. Default to None.
        residual: (bool)
            Whether to add skip connection to the output. Default to True.
        dropout: (float, None):
            Dropout applied to the output of the layer. Default to None.
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

    References:
        .. [#] https://arxiv.org/pdf/2003.00982.pdf
        .. [#] https://arxiv.org/pdf/1810.00826.pdf
        .. [#] https://arxiv.org/pdf/1905.12265.pdf
    '''

    def __init__(
        self,
        units: Optional[int] = None,
        apply_relu_activation: bool = False,
        self_projection: bool = True,
        normalization: Union[None, str, bool] = None,
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

        self.normalization = normalization
        self.activation = activations.get('relu')
        self.apply_self_projection = self_projection
        self.apply_relu_activation = apply_relu_activation

    def _build(self, graph_tensor_spec: GraphTensor.Spec) -> None:

        self.projection_1 = self.get_dense(self.units)
        self.projection_2 = self.get_dense(self.units)
        self.epsilon = self.add_weight(
            shape=(),
            initializer='zeros',
            regularizer=None,
            trainable=True,
            name='epsilon')

        node_dim = graph_tensor_spec.node_feature.shape[-1]
        if self.use_edge_features:
            edge_dim = graph_tensor_spec.edge_feature.shape[-1]
            if edge_dim != node_dim:
                self.edge_projection = self.get_dense(node_dim)

        if self.normalization:
            if str(self.normalization).startswith('batch'):
                self.normalization_fn = layers.BatchNormalization()
            else:
                self.normalization_fn = layers.LayerNormalization()

        if self.apply_self_projection:
           self.self_projection = self.get_dense(self.units)

    def _call(self, tensor: GraphTensor) -> GraphTensor:

        if not self.use_edge_features:
            node_feature_aggregated = gnn_ops.propagate_node_features(
                node_feature=tensor.node_feature,
                edge_src=tensor.edge_src,
                edge_dst=tensor.edge_dst,
            )
        else:
            node_feature_src = tf.gather(tensor.node_feature, tensor.edge_src)
            if hasattr(self, 'edge_projection'):
                edge_feature_updated = self.edge_projection(tensor.edge_feature)
                tensor = tensor.update({'edge_feature': edge_feature_updated})
            node_feature_src += tensor.edge_feature
            if self.apply_relu_activation:
                node_feature_src = tf.nn.relu(node_feature_src)
            node_feature_aggregated = tf.math.unsorted_segment_sum(
                node_feature_src, tensor.edge_dst, tf.shape(tensor.node_feature)[0])
            
        node_feature = (
            (1 + self.epsilon) * tensor.node_feature + node_feature_aggregated)

        node_feature = self.projection_1(node_feature)

        if self.apply_self_projection:
            node_feature += self.self_projection(tensor.node_feature)

        if self.normalization:
            node_feature = self.normalization_fn(node_feature)

        node_feature = self.activation(node_feature)

        node_feature = self.projection_2(node_feature)

        return tensor.update({'node_feature': node_feature})

    def get_config(self):
        base_config = super().get_config()
        config = {
            'self_projection': self.apply_self_projection,
            'apply_relu_activation': self.apply_relu_activation
        }
        base_config.update(config)
        return base_config
