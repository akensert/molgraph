import tensorflow as tf
from tensorflow import keras

from keras import initializers
from keras import regularizers
from keras import constraints
from keras import layers

from typing import Optional
from typing import Callable
from typing import Union

from molgraph.internal import register_keras_serializable 

from molgraph.tensors.graph_tensor import GraphTensor

from molgraph.layers import gnn_layer


@register_keras_serializable(package='molgraph')
class GMMConv(gnn_layer.GNNLayer):

    '''Multi-head graph gaussian mixture layer (MoNet)

    Implementation is based on Dwivedi et al. (2022) [#]_ and 
    Monti et al. (2016) [#]_.

    Example usage:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     sizes=[2, 3],
    ...     node_feature=[[1., 0.], [1., 0.], [1., 0.], [1., 0.], [0., 1.]],
    ...     edge_src=[1, 0, 3, 4, 2, 4, 3, 2],
    ...     edge_dst=[0, 1, 2, 2, 3, 3, 4, 4],
    ... )
    >>> gnn_model = tf.keras.Sequential([
    ...     molgraph.layers.GMMConv(units=16),
    ...     molgraph.layers.GMMConv(units=16),
    ...     molgraph.layers.GMMConv(units=16),
    ...     molgraph.layers.Readout(),
    ... ])
    >>> gnn_model(graph_tensor).shape
    TensorShape([2, 16])

    Args:
        units (int, None):
            Number of output units.
        num_kernels (int):
            Number of attention heads. Default to 8.
        merge_mode (str):
            The strategy for merging the heads. Either of 'concat', 'sum',
            'mean' or None. If set to None, 'mean' is used. Default to 'sum'.
        pseudo_coord_dim (int):
            The dimension of the pseudo coordinate of the Gaussian kernel.
            Default to 2.
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

    .. [#] https://arxiv.org/pdf/2003.00982.pdf
    .. [#] https://arxiv.org/pdf/1611.08402.pdf

    '''

    def __init__(
        self,
        units: Optional[int] = None,
        num_kernels: int = 8,
        merge_mode: Optional[str] = 'sum',
        pseudo_coord_dim=2,
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

        self.num_kernels = num_kernels
        self.merge_mode = merge_mode
        self.pseudo_coord_dim = pseudo_coord_dim
        self.apply_self_projection = self_projection

    def _build(self, graph_tensor_spec: GraphTensor.Spec) -> None:

        if self.merge_mode == 'concat':
            if not self.units or (self.units % self.num_kernels != 0):
                raise ValueError(
                    "`merge_mode` was set to `concat` and hence " +
                    " need `units` to be divisble by `num_heads`")
            self.units //= self.num_kernels

        self.node_projection = self.get_einsum_dense(
            'ij,jkh->ihk', (self.num_kernels, self.units))

        if self.apply_self_projection:
            self.self_projection = self.get_einsum_dense(
                'ij,jkh->ihk', (self.num_kernels, self.units))

        self.pseudo_coord_projection = layers.Dense(
            self.pseudo_coord_dim, activation='tanh')

        self.mu = self.add_weight(
            shape=(1, self.num_kernels, self.pseudo_coord_dim),
            initializer='random_normal',
            name='mu')

        self.sigma_inv = self.add_weight(
            shape=(1, self.num_kernels, self.pseudo_coord_dim),
            initializer='ones',
            name='sigma_inv')

        if self.merge_mode == 'concat':
            self.units *= self.num_kernels

    def _call(self, tensor: GraphTensor) -> GraphTensor:

        if self.apply_self_projection:
            node_feature_residual = self.self_projection(tensor.node_feature)
        else:
            node_feature_residual = None

        node_feature = self.node_projection(tensor.node_feature)

        edge_weights = self.compute_edge_weights(
            tensor.edge_dst, tensor.edge_src)

        tensor = tensor.update({
            'node_feature': node_feature, 'edge_weight': edge_weights})

        return tensor.propagate(
            reduction=self.merge_mode,
            residual=node_feature_residual)

    def compute_edge_weights(self, edge_dst, edge_src, clip_values=(-5, 5)):
        """Computes edge weights via Gaussian kernels"""

        def true_fn(edge_dst, edge_src, clip_values):
            """If edges exist, call this function"""
            # Compute self/neighbor degree-tuples (deg(i), deg(j))
            degree = tf.math.bincount(edge_dst)
            degree = tf.cast(degree, tf.float32)
            degree = tf.where(degree == 0.0, 1.0, degree) ** -(1/2)
            degree = tf.stack([
                tf.gather(degree, edge_src),
                tf.gather(degree, edge_dst),
            ], axis=1)

            # (n_edges, 2) @ (2, 2) -> (n_edges, 2)
            pseudo_coord = self.pseudo_coord_projection(degree)
            pseudo_coord = tf.expand_dims(pseudo_coord, 1)

            # (n_edges, 1, dim), (1, kernels, dim)
            edge_weights = -1/2 * (pseudo_coord - self.mu) ** 2
            # (n_edges, kernels, dim)
            edge_weights *= (self.sigma_inv ** 2)
            # (n_edges, kernels, dim)
            edge_weights = tf.reduce_sum(edge_weights, axis=-1, keepdims=True)
            # (n_edges, kernels, 1)
            return tf.exp(tf.clip_by_value(edge_weights, *clip_values))

        def false_fn():
            """If no edges exist, call this function"""
            return tf.zeros([0, self.num_kernels, 1], dtype=tf.float32)

        return tf.cond(
            tf.greater(tf.shape(edge_dst)[0], 0),
            lambda: true_fn(edge_dst, edge_src, clip_values),
            lambda: false_fn()
        )

    def get_config(self):
        base_config = super().get_config()
        config = {
            'num_kernels': self.num_kernels,
            'merge_mode': self.merge_mode,
            'pseudo_coord_dim': self.pseudo_coord_dim,
            'self_projection': self.apply_self_projection,
        }
        base_config.update(config)
        return base_config
