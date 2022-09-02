import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.keras import activations
from tensorflow.keras import layers
import math

from typing import Optional
from typing import Callable
from typing import Union
from typing import Tuple

from molgraph.tensors.graph_tensor import GraphTensor
from molgraph.layers.base import BaseLayer
from molgraph.layers.ops import compute_edge_weights_from_degrees
from molgraph.layers.ops import propagate_node_features



@keras.utils.register_keras_serializable(package='molgraph')
class MPNNConv(BaseLayer):

    """Message passing neural network layer (MPNN)

    Implementation is based on Gilmer et al. (2017) [#]_. In contrast to Gilmer
    et al. this implementation does not use weight tying; for neither the message
    function nor the update function. Furthermore, instead of the GRU-based
    update function, here, a simple fully-connected (dense) layer is used.

    For an MPNN more similar to Gilmer et al., see ``molgraph.models.MPNN``.

    **Examples:**

    Inputs a ``GraphTensor`` encoding (two) subgraphs:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     data={
    ...         'edge_dst': [[0, 1], [0, 0, 1, 1, 2, 2]],
    ...         'edge_src': [[1, 0], [1, 2, 0, 2, 1, 0]],
    ...         'node_feature': [
    ...             [[1.0, 0.0], [1.0, 0.0]],
    ...             [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    ...         ],
    ...         'edge_feature': [
    ...             [[1.0, 0.0], [0.0, 1.0]],
    ...             [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0],
    ...              [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    ...         ],
    ...     }
    ... )
    >>> # Build a model with MPNNConv
    >>> gnn_model = tf.keras.Sequential([
    ...     tf.keras.Input(type_spec=graph_tensor.unspecific_spec),
    ...     molgraph.layers.MPNNConv(16, activation='relu'),
    ...     molgraph.layers.MPNNConv(16, activation='relu')
    ... ])
    >>> gnn_model.output_shape
    (None, None, 16)

    Inputs a ``GraphTensor`` encoding a single disjoint graph:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     data={
    ...         'edge_dst': [0, 1, 2, 2, 3, 3, 4, 4],
    ...         'edge_src': [1, 0, 3, 4, 2, 4, 3, 2],
    ...         'node_feature': [
    ...             [1.0, 0.0],
    ...             [1.0, 0.0],
    ...             [1.0, 0.0],
    ...             [1.0, 0.0],
    ...             [0.0, 1.0]
    ...         ],
    ...         'graph_indicator': [0, 0, 1, 1, 1],
    ...         'edge_feature': [
    ...             [1.0, 0.0],
    ...             [0.0, 1.0],
    ...             [0.0, 1.0],
    ...             [0.0, 1.0],
    ...             [1.0, 0.0],
    ...             [0.0, 1.0],
    ...             [1.0, 0.0],
    ...             [0.0, 1.0]
    ...         ],
    ...     }
    ... )
    >>> # Build a model with MPNNConv
    >>> gnn_model = tf.keras.Sequential([
    ...     tf.keras.Input(type_spec=graph_tensor.unspecific_spec),
    ...     molgraph.layers.MPNNConv(16, activation='relu'),
    ...     molgraph.layers.MPNNConv(16, activation='relu')
    ... ])
    >>> gnn_model.output_shape
    (None, 16)

    Args:
        units (int, None):
            Number of output units.
        use_edge_features (bool):
            Whether or not to use edge features. Default to True.
        self_projection (bool):
            Whether to apply self projection. Default to True.
        batch_norm: (bool):
            Whether to apply batch normalization to the output. Default to True.
        residual: (bool)
            Whether to add skip connection to the output. Default to True.
        dropout: (float, None):
            Dropout applied to the output of the layer. Default to None.
        update_activation (tf.keras.activations.Activation, callable, str, None):
            Activation function used for the update function. Default to None.
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

    References:
        .. [#] https://arxiv.org/pdf/1704.01212.pdf

    """

    def __init__(
        self,
        units: Optional[int] = None,
        use_edge_features: bool = True,
        self_projection: bool = True,
        batch_norm: bool = True,
        residual: bool = True,
        dropout: Optional[float] = None,
        update_activation: Union[None, str, Callable[[tf.Tensor], tf.Tensor]] = None,
        activation: Union[None, str, Callable[[tf.Tensor], tf.Tensor]] = 'relu',
        use_bias: bool = True,
        kernel_initializer: Union[
            str, initializers.Initializer
        ] = initializers.TruncatedNormal(stddev=0.005),
        bias_initializer: Union[
            str, initializers.Initializer
        ] = initializers.Constant(0.),
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_regularizer: Optional[regularizers.Regularizer] = None,
        activity_regularizer: Optional[regularizers.Regularizer] = None,
        kernel_constraint: Optional[constraints.Constraint] = None,
        bias_constraint: Optional[constraints.Constraint] = None,
        **kwargs
    ):
        super().__init__(
            units=units,
            batch_norm=batch_norm,
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

        self.use_edge_features = use_edge_features
        self.apply_self_projection = self_projection
        self.update_activation = update_activation

    def subclass_build(
        self,
        node_feature_shape: tf.TensorShape,
        edge_feature_shape: Optional[tf.TensorShape] = None
    ) -> None:
        self.message_projection = self.get_dense(self.units * self.units)
        self.update_step = keras.layers.Dense(
            self.units, activation=self.update_activation)

        if (
            self.units != node_feature_shape[-1] and
            not hasattr(self, 'node_resample')
        ):
            self.node_resample = self.get_dense(self.units)

        if self.apply_self_projection:
            self.self_projection = self.get_dense(self.units)

    def subclass_call(self, tensor: GraphTensor) -> GraphTensor:

        if hasattr(self, 'node_resample'):
            tensor = tensor.update({
                'node_feature': self.node_resample(tensor.node_feature)})

        # MPNN requires edge features, if edge features do not exist,
        # we force edge features by initializing them as ones vector
        if not hasattr(tensor, 'edge_feature') or not self.use_edge_features:
            tensor = tensor.update({
                'edge_feature': tf.ones(
                    shape=[tf.shape(tensor.edge_dst)[0], 1], dtype=tf.float32)})

        node_feature_aggregated = message_step(
            node_feature=tensor.node_feature,
            edge_feature=tensor.edge_feature,
            edge_dst=tensor.edge_dst,
            edge_src=tensor.edge_src,
            projection=self.message_projection)

        if self.apply_self_projection:
            node_feature_aggregated += self.self_projection(tensor.node_feature)

        node_feature_update = self.update_step(
            tf.concat([tensor.node_feature, node_feature_aggregated], axis=1))

        return tensor.update({'node_feature': node_feature_update})

    def get_config(self):
        base_config = super().get_config()
        config = {
            'use_edge_features': self.use_edge_features,
            'self_projection': self.apply_self_projection,
            'update_activation': self.update_activation
        }
        base_config.update(config)
        return base_config


def message_step(
    node_feature: tf.Tensor,
    edge_feature: tf.Tensor,
    edge_dst: tf.Tensor,
    edge_src: tf.Tensor,
    projection: keras.layers.Dense,
) -> tf.Tensor:
    '''Performs a message passing step.

    Args:
        node_feature (tf.Tensor):
            Node features; field of GraphTensor.
        edge_feature (tf.Tensor):
            Edge features; field of GraphTensor.
        edge_dst (tf.Tensor):
            Destination node indices; field of GraphTensor.
        edge_src (tf.Tensor):
            Source node indices; field of GraphTensor.
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
