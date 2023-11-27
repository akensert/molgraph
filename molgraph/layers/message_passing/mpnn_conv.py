import tensorflow as tf
from tensorflow import keras

from keras import initializers
from keras import regularizers
from keras import constraints

import math

from typing import Optional
from typing import Callable
from typing import Union

from molgraph.internal import register_keras_serializable 

from molgraph.tensors.graph_tensor import GraphTensor

from molgraph.layers import gnn_layer


@register_keras_serializable(package='molgraph')
class MPNNConv(gnn_layer.GNNLayer):

    """Message passing neural network layer (MPNN)

    Implementation is based on Gilmer et al. (2017) [#]_. In contrast to Gilmer
    et al. this implementation does not use weight tying by default; for neither the 
    message function nor the update function. Furthermore, instead of the GRU-based
    update function, here, a simple fully-connected (dense) layer can be used too.
    Though default is GRU. 

    Example usage:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     sizes=[2, 3],
    ...     node_feature=[[1., 0.], [1., 0.], [1., 0.], [1., 0.], [0., 1.]],
    ...     edge_feature=[[1., 0.], [0., 1.], [0., 1.], [0., 1.], 
    ...                   [1., 0.], [0., 1.], [1., 0.], [0., 1.]],
    ...     edge_src=[1, 0, 3, 4, 2, 4, 3, 2],
    ...     edge_dst=[0, 1, 2, 2, 3, 3, 4, 4],
    ... )
    >>> gnn_model = tf.keras.Sequential([
    ...     molgraph.layers.MPNNConv(units=16),
    ...     molgraph.layers.MPNNConv(units=16),
    ...     molgraph.layers.MPNNConv(units=16),
    ...     molgraph.layers.Readout(),
    ... ])
    >>> gnn_model(graph_tensor).shape
    TensorShape([2, 16])

    Pass the same GRU cell to each layer to perform weight sharing:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     sizes=[2, 3],
    ...     node_feature=[[1., 0.], [1., 0.], [1., 0.], [1., 0.], [0., 1.]],
    ...     edge_feature=[[1., 0.], [0., 1.], [0., 1.], [0., 1.], 
    ...                   [1., 0.], [0., 1.], [1., 0.], [0., 1.]],
    ...     edge_src=[1, 0, 3, 4, 2, 4, 3, 2],
    ...     edge_dst=[0, 1, 2, 2, 3, 3, 4, 4],
    ... )
    >>> gru_cell = tf.keras.layers.GRUCell(16)
    >>> gnn_model = tf.keras.Sequential([
    ...     molgraph.layers.MPNNConv(units=16, update_fn=gru_cell),
    ...     molgraph.layers.MPNNConv(units=16, update_fn=gru_cell),
    ...     molgraph.layers.MPNNConv(units=16, update_fn=gru_cell),
    ...     molgraph.layers.Readout(),
    ... ])
    >>> gnn_model(graph_tensor).shape
    TensorShape([2, 16])

    Args:
        units (int, None):
            Number of output units.
        update_mode (bool):
            Specify what type of update will be performed. Either 'dense' or 'gru'. 
            Default to 'gru'.
        update_fn (tf.keras.layers.GRUCell, tf.keras.layers.Dense, None):
            Optionally pass update function (GRUCell or Dense) for weight-tying 
            of the update step (GRU step or Dense step). Default to None.
        self_projection (bool):
            Whether to apply self projection. Default to True.
        normalization: (None, str, bool):
            Whether to apply layer normalization to the output. If batch 
            normalization is desired, pass 'batch_norm'. Default to None.
        residual: (bool)
            Whether to add skip connection to the output. Default to True.
        dropout: (float, None):
            Dropout applied to the output of the layer. Default to None.
        update_activation (tf.keras.activations.Activation, callable, str, None):
            Activation function used for the update function. Only relevant if 'dense' is passed
            to the `update_mode` argument. Default to None.
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
        .. [#] https://arxiv.org/pdf/1704.01212.pdf

    """

    def __init__(
        self,
        units: Optional[int] = None,
        update_mode: str = 'gru',
        update_fn: Optional[Union[
            keras.layers.GRUCell, keras.layers.Dense]] = None,
        self_projection: bool = True,
        normalization: Union[None, str, bool] = None,
        residual: bool = True,
        dropout: Optional[float] = None,
        update_activation: Union[None, str, Callable[[tf.Tensor], tf.Tensor]] = None,
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
            use_edge_features=kwargs.pop('use_edge_features', True),
            **kwargs)

        self.update_fn = update_fn
        self.update_mode = update_mode.lower()
        self.apply_self_projection = self_projection
        if update_activation is None:
            if self.update_mode.startswith('gru'):
                self.update_activation = 'tanh'
            else:
                self.update_activation = 'linear'
        else:
            self.update_activation = update_activation

    def _build(self, graph_tensor_spec: GraphTensor.Spec) -> None:
     
        self.message_projection = self.get_dense(self.units * self.units)

        if self.update_fn is None:
            if self.update_mode.startswith('gru'):
                common_kwargs = self._get_common_kwargs()
                common_kwargs['kernel_initializer'] = 'glorot_uniform'
                common_kwargs['recurrent_initializer'] = 'orthogonal'
                common_kwargs.pop('activity_regularizer', None)
                self.update_fn = tf.keras.layers.GRUCell(
                    self.units, 
                    activation=self.update_activation,
                    **common_kwargs)
            else:
                self.update_fn = self.get_dense(
                    self.units, self.update_activation)
        
        node_dim = graph_tensor_spec.node_feature.shape[-1]
        if self.units != node_dim and not hasattr(self, 'node_resample'):
            self.node_resample = self.get_dense(self.units)

        if self.apply_self_projection:
            self.self_projection = self.get_dense(self.units)
        else:
            self.self_projection = None

    def _call(self, tensor: GraphTensor) -> GraphTensor:

        if hasattr(self, 'node_resample'):
            tensor = tensor.update({
                'node_feature': self.node_resample(tensor.node_feature)})

        # MPNN requires edge features, if edge features do not exist,
        # we force edge features by initializing them as ones vector
        if tensor.edge_feature is None:
            tensor = tensor.update({
                'edge_feature': tf.ones(
                    shape=[tf.shape(tensor.edge_src)[0], 1], dtype=tf.float32)})

        node_feature_aggregated = message_step(
            node_feature=tensor.node_feature,
            edge_feature=tensor.edge_feature,
            edge_src=tensor.edge_src,
            edge_dst=tensor.edge_dst,
            projection=self.message_projection)

        node_feature_update = update_step(
            node_feature=node_feature_aggregated,
            node_feature_prev=tensor.node_feature,
            update_projection=self.update_fn,
            self_projection=self.self_projection)
        
        return tensor.update({'node_feature': node_feature_update})

    def get_config(self):
        base_config = super().get_config()
        config = {
            'update_mode': self.update_mode,
            'update_fn': tf.keras.layers.serialize(self.update_fn),
            'self_projection': self.apply_self_projection,
            'update_activation': self.update_activation
        }
        base_config.update(config)
        return base_config


def update_step(
    node_feature: tf.Tensor, 
    node_feature_prev: tf.Tensor,
    update_projection: Union[keras.layers.GRUCell, keras.layers.Dense],
    self_projection: keras.layers.Dense,
) -> tf.Tensor:
    if self_projection:
        node_feature += self_projection(node_feature_prev)
    if isinstance(update_projection, keras.layers.Dense):
        node_feature = update_projection(
            tf.concat([node_feature_prev, node_feature], axis=1))
    else:
        node_feature, _ = update_projection(
            inputs=node_feature,
            states=node_feature_prev)
    return node_feature


def message_step(
    node_feature: tf.Tensor,
    edge_feature: tf.Tensor,
    edge_src: tf.Tensor,
    edge_dst: tf.Tensor,
    projection: keras.layers.Dense,
) -> tf.Tensor:
    '''Performs a message passing step.

    Args:
        node_feature (tf.Tensor):
            Node features; component of GraphTensor.
        edge_feature (tf.Tensor):
            Edge features; component of GraphTensor.
        edge_src (tf.Tensor):
            Source node indices; component of GraphTensor.
        edge_dst (tf.Tensor):
            Destination node indices; component of GraphTensor.
        projection (keras.layers.Dense):
            Dense layer that transforms edge features.

    Returns (tf.Tensor):
        Returns updated (aggregated) node features.
    '''
    output_units = int(math.sqrt(projection.units))
    # Apply linear transformation to edge features
    edge_feature = projection(edge_feature)
    # Reshape edge features to match source nodes' features
    edge_feature = tf.reshape(edge_feature, (-1, output_units, output_units))
    # Obtain source nodes' features (1-hop neighbor nodes)
    node_feature_src = tf.expand_dims(tf.gather(node_feature, edge_src), -1)
    # Apply edge features (obtain messages to be passed to destination nodes)
    messages = tf.squeeze(tf.matmul(edge_feature, node_feature_src), -1)
    # Send messages to destination nodes
    return tf.math.unsorted_segment_sum(
        data=messages,
        segment_ids=edge_dst,
        num_segments=tf.shape(node_feature)[0])
