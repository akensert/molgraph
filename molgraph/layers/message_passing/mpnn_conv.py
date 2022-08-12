import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.keras import activations
from tensorflow.keras import layers

from typing import Optional
from typing import Callable
from typing import Union
from typing import Tuple

from molgraph.tensors.graph_tensor import GraphTensor
from molgraph.layers.base import _BaseLayer
from molgraph.layers.ops import compute_edge_weights_from_degrees
from molgraph.layers.ops import propagate_node_features


@keras.utils.register_keras_serializable(package='molgraph')
class MPNNConv(_BaseLayer):

    """Message passing (neural network) layer based on  Gilmer et al. [#]_.

    References
    ----------
    .. [#] Gilmer et al. https://arxiv.org/pdf/1704.01212.pdf

    """

    def __init__(
        self,
        units: Optional[int] = None,
        use_edge_features: bool = True,
        self_projection: bool = True,
        batch_norm: bool = True,
        residual: bool = True,
        dropout: Optional[float] = None,
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
        self.residual = residual
        self.apply_self_projection = self_projection

    def subclass_build(
        self,
        node_feature_shape: tf.TensorShape,
        edge_feature_shape: Optional[tf.TensorShape] = None
    ) -> None:
        self.message_projection = self.get_dense(self.units * self.units)
        self.update_step = keras.layers.GRUCell(self.units)

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

        # Aggregate node states from neighbors
        node_feature_aggregated = self.message_step(tensor)

        # Update aggregated node states via a step of GRU
        node_feature_update, _ = self.update_step(
            node_feature_aggregated, tensor.node_feature)

        return tensor.update({'node_feature': node_feature_update})

    def message_step(self, tensor: GraphTensor) -> tf.Tensor:

        # MPNN requires edge features, if edge features do not exist,
        # we force edge features by initializing a ones vector
        if not hasattr(tensor, 'edge_feature') or not self.use_edge_features:
            tensor = tensor.update({
                'edge_feature': tf.ones(
                    shape=[tf.shape(tensor.edge_dst)[0], 1], dtype=tf.float32)})

        # Apply linear transformation to edge features
        edge_feature = self.message_projection(tensor.edge_feature)

        # Reshape edge features for neighborhood aggregation later
        edge_feature = tf.reshape(edge_feature, (-1, self.units, self.units))

        # Obtain node states of neighbors
        node_feature_src = tf.gather(tensor.node_feature, tensor.edge_src)
        node_feature_src = tf.expand_dims(node_feature_src, axis=-1)

        # Apply transformation followed by neighborhood aggregation
        node_feature_src = tf.matmul(edge_feature, node_feature_src)
        node_feature_src = tf.squeeze(node_feature_src, axis=-1)

        node_feature_updated = tf.math.unsorted_segment_sum(
            data=node_feature_src,
            segment_ids=tensor.edge_dst,
            num_segments=tf.shape(tensor.node_feature)[0])

        if self.apply_self_projection:
            node_feature_updated += self.self_projection(tensor.node_feature)

        return node_feature_updated

    def get_config(self):
        base_config = super().get_config()
        config = {
            'use_edge_features': self.use_edge_features,
            'self_projection': self.apply_self_projection,
        }
        base_config.update(config)
        return base_config
