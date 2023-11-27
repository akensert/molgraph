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


@register_keras_serializable(package='molgraph')
class GraphSageConv(gnn_layer.GNNLayer):

    '''Graph sage convolution layer (GraphSage)

    Implementation is based on Hamilton et al. (2018) [#]_ and
    Dwivedi et al. (2022) [#]_.

    Example usage:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     sizes=[2, 3],
    ...     node_feature=[[1., 0.], [1., 0.], [1., 0.], [1., 0.], [0., 1.]],
    ...     edge_src=[1, 0, 3, 4, 2, 4, 3, 2],
    ...     edge_dst=[0, 1, 2, 2, 3, 3, 4, 4],
    ... )
    >>> gnn_model = tf.keras.Sequential([
    ...     molgraph.layers.GraphSageConv(units=16),
    ...     molgraph.layers.GraphSageConv(units=16),
    ...     molgraph.layers.GraphSageConv(units=16),
    ...     molgraph.layers.Readout(),
    ... ])
    >>> gnn_model(graph_tensor).shape
    TensorShape([2, 16])

    Args:
        units (int, None):
            Number of output units.
        aggregation_mode (str):
            Type of neighborhood aggregation to be performed. Either of 'max',
            'lstm', 'mean', 'sum'. Default to 'mean'.
        normalize (bool):
            Whether l2 normalization should be performed on the updated node
            features. Default to True.
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

    References:
        .. [#] https://arxiv.org/pdf/1706.02216.pdf
        .. [#] https://arxiv.org/pdf/2003.00982.pdf

    '''

    def __init__(
        self,
        units: Optional[int] = None,
        aggregation_mode: str = 'mean',
        normalize: bool = True,
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
            use_edge_features=kwargs.pop('use_edge_features', False),
            **kwargs)

        self.aggregation_mode = aggregation_mode
        self.normalize = normalize if not normalization else False
        self.apply_self_projection = self_projection
        self.activation = activations.get('relu')

    def _build(self, graph_tensor_spec: GraphTensor.Spec) -> None:

        if self.aggregation_mode == 'max':
            self.node_src_projection = self.get_dense(self.units)
        elif self.aggregation_mode == 'lstm':
            self.lstm = layers.LSTM(self.units)

        if self.apply_self_projection:
            self.self_projection = self.get_dense(self.units)

        self.node_projection = self.get_dense(self.units)

    def _call(self, tensor: GraphTensor) -> GraphTensor:
        
        node_feature_residual = tensor.node_feature

        if self.aggregation_mode == 'lstm':
            node_feature = self.lstm_aggregate(
                tensor.node_feature, tensor.edge_src, tensor.edge_dst)
        else:
            if self.aggregation_mode == 'max':
                node_feature = self.node_src_projection(tensor.node_feature)
                node_feature = self.activation(node_feature)
            else:
                node_feature = tensor.node_feature

            tensor = tensor.update({'node_feature': node_feature})
            tensor = tensor.propagate(mode=self.aggregation_mode)
            
        node_feature = tf.concat([
            tensor.node_feature, node_feature_residual], axis=-1)

        node_feature = self.node_projection(node_feature)

        if self.apply_self_projection:
            node_feature += self.self_projection(node_feature_residual)

        if self.normalize:
            node_feature = self.activation(node_feature)
            node_feature = tf.math.l2_normalize(node_feature, axis=1)

        return tensor.update({'node_feature': node_feature})

    def lstm_aggregate(self, node_feature, edge_src, edge_dst):

        def true_fn(node_feature, edge_src, edge_dst):
            'If edges exist, call this function'

            # Get number of nodes
            num_nodes = tf.shape(node_feature)[0]

            # Shuffle neighbor (source) nodes of the destination nodes.
            random_indices = tf.random.shuffle(tf.range(tf.shape(edge_src)[0]))
            edge_dst = tf.gather(edge_dst, random_indices)
            edge_src = tf.gather(edge_src, random_indices)
            sorted_indices = tf.argsort(edge_dst)
            edge_dst = tf.gather(edge_dst, sorted_indices)
            edge_src = tf.gather(edge_src, sorted_indices)

            # Gather source nodes followed by a partitioning of destination nodes
            node_feature = tf.RaggedTensor.from_value_rowids(
                tf.gather(node_feature, edge_src), edge_dst, num_nodes)
            node_feature = node_feature.to_tensor()
            # Pass to lstm for update
            node_feature = self.lstm(node_feature)

            return node_feature

        def false_fn(node_feature):
            """If no edges exist, call this function"""
            return tf.zeros(
                shape=(tf.shape(node_feature)[0], self.units),
                dtype=node_feature.dtype)

        return tf.cond(
            tf.greater(tf.shape(edge_src)[0], 0),
            lambda: true_fn(node_feature, edge_src, edge_dst),
            lambda: false_fn(node_feature)
        )

    def get_config(self):
        base_config = super().get_config()
        config = {
            'aggregation_mode': self.aggregation_mode,
            'normalize': self.normalize,
            'self_projection': self.apply_self_projection,
        }
        base_config.update(config)
        return base_config
