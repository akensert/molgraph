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
from molgraph.layers.base import BaseLayer
from molgraph.layers.ops import propagate_node_features
from molgraph.layers.ops import reduce_features



@keras.utils.register_keras_serializable(package='molgraph')
class GMMConv(BaseLayer):

    '''Multi-head graph gaussian mixture layer (MoNet)

    Implementation is based on Dwivedi et al. (2022) [#]_ and Monti et al. (2016) [#]_.

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
    ...     }
    ... )
    >>> # Build a model with GMMConv
    >>> gnn_model = tf.keras.Sequential([
    ...     tf.keras.Input(type_spec=graph_tensor.unspecific_spec),
    ...     molgraph.layers.GMMConv(16, activation='relu'),
    ...     molgraph.layers.GMMConv(16, activation='relu')
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
    ...     }
    ... )
    >>> # Build a model with GMMConv
    >>> gnn_model = tf.keras.Sequential([
    ...     tf.keras.Input(type_spec=graph_tensor.unspecific_spec),
    ...     molgraph.layers.GMMConv(16, activation='relu'),
    ...     molgraph.layers.GMMConv(16, activation='relu')
    ... ])
    >>> gnn_model.output_shape
    (None, 16)

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
        batch_norm: (bool):
            Whether to apply batch normalization to the output. Default to True.
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

        self.num_kernels = num_kernels
        self.merge_mode = merge_mode
        self.pseudo_coord_dim = pseudo_coord_dim
        self.apply_self_projection = self_projection

    def subclass_build(
        self,
        node_feature_shape: Optional[tf.TensorShape] = None,
        edge_feature_shape: Optional[tf.TensorShape] = None
    ) -> None:

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

    def subclass_call(self, tensor: GraphTensor) -> GraphTensor:

        node_feature = self.node_projection(tensor.node_feature)

        edge_weights = self.compute_edge_weights(
            tensor.edge_dst, tensor.edge_src)

        node_feature = propagate_node_features(
            node_feature=node_feature,
            edge_dst=tensor.edge_dst,
            edge_src=tensor.edge_src,
            edge_weight=edge_weights)

        if self.apply_self_projection:
            node_feature += self.self_projection(tensor.node_feature)

        node_feature = reduce_features(
            feature=node_feature,
            mode=self.merge_mode,
            output_units=self.units)

        return tensor.update({'node_feature': node_feature})

    def compute_edge_weights(self, edge_dst, edge_src, clip_values=(-5, 5)):
        """Computes edge weights via Gaussian kernels"""

        def true_fn(edge_dst, edge_src, clip_values):
            """If edges exist, call this function"""
            # Compute self/neighbor degree-tuples (deg(i), deg(j))
            degree = tf.math.bincount(edge_dst)
            degree = tf.cast(degree, tf.float32)
            degree = tf.where(degree == 0.0, 1.0, degree) ** -(1/2)
            degree = tf.stack([
                tf.gather(degree, edge_dst),
                tf.gather(degree, edge_src)], axis=1)

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
