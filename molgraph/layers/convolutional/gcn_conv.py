import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.keras import activations

from typing import Optional
from typing import Callable
from typing import Union
from typing import Tuple

from molgraph.tensors.graph_tensor import GraphTensor
from molgraph.layers.base import BaseLayer
from molgraph.layers.ops import compute_edge_weights_from_degrees
from molgraph.layers.ops import propagate_node_features



@keras.utils.register_keras_serializable(package='molgraph')
class GCNConv(BaseLayer):

    """Graph convolutional layer (GCN).

    Implementation is based on Kipf et al. (2017) [#]_, Dwivedi et al. (2022) [#]_,
    and, for RGCN, Schlichtkrull et al. (2017) [#]_.

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
    >>> # Build a model with GCNConv
    >>> gnn_model = tf.keras.Sequential([
    ...     tf.keras.Input(type_spec=graph_tensor.unspecific_spec),
    ...     molgraph.layers.GCNConv(16, activation='relu'),
    ...     molgraph.layers.GCNConv(16, activation='relu')
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
    >>> # Build a model with GCNConv
    >>> gnn_model = tf.keras.Sequential([
    ...     tf.keras.Input(type_spec=graph_tensor.unspecific_spec),
    ...     molgraph.layers.GCNConv(16, activation='relu'),
    ...     molgraph.layers.GCNConv(16, activation='relu')
    ... ])
    >>> gnn_model.output_shape
    (None, 16)

    Args:
        units (int, None):
            Number of output units.
        use_edge_features (bool):
            Whether or not to use edge features. Default to False.
        degree_normalization (str, None):
            The strategy for computing edge weights from degrees. Either of
            'symmetric', 'row' or None. If None, 'row' is used. Default to 'symmetric'.
        num_bases (int, None):
            Number of bases to use for basis decomposition. Only relevant if
            use_edge_features is True. Default to None.
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
        .. [#] https://arxiv.org/pdf/1609.02907.pdf
        .. [#] https://arxiv.org/pdf/2003.00982.pdf
        .. [#] https://arxiv.org/pdf/1703.06103.pdf
    """

    def __init__(
        self,
        units: Optional[int] = None,
        use_edge_features: bool = False,
        degree_normalization: Optional[str] = 'symmetric',
        num_bases: Optional[int] = None,
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
        kwargs.pop("use_bias", None)
        super().__init__(
            units=units,
            batch_norm=batch_norm,
            residual=residual,
            dropout=dropout,
            activation=activation,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

        self.use_edge_features = use_edge_features
        self.degree_normalization = degree_normalization
        self.num_bases = (
            int(num_bases) if isinstance(num_bases, (int, float)) else 0)
        self.apply_self_projection = self_projection
        self.use_bias = use_bias

    def subclass_build(
        self,
        node_feature_shape: Optional[tf.TensorShape] = None,
        edge_feature_shape: Optional[tf.TensorShape] = None
    ) -> None:

        kernel_shape = [self.node_dim, self.units]
        bias_shape = [self.units]

        if edge_feature_shape is None:
            self.use_edge_features = False

        if self.use_edge_features:

            edge_dim = edge_feature_shape[-1]

            if self.num_bases > 0 and self.num_bases < edge_dim:
                num_bases = self.num_bases
                self.kernel_decomp = self.get_kernel(
                    (edge_dim, num_bases), name='kernel_decomp')
            else:
                num_bases = edge_dim

            kernel_shape += [num_bases]
            bias_shape += [edge_dim]

        self.kernel = self.get_kernel(kernel_shape, name='kernel')

        if self.use_bias:
            self.bias = self.get_bias(bias_shape, name='bias')

        if self.apply_self_projection:
            self.self_projection = self.get_dense(self.units)

    def subclass_call(self, tensor: GraphTensor) -> GraphTensor:

        if self.use_edge_features:
            node_feature = tf.expand_dims(tensor.node_feature, axis=-1)
        else:
            node_feature = tensor.node_feature

        edge_weight = compute_edge_weights_from_degrees(
            edge_dst=tensor.edge_dst,
            edge_src=tensor.edge_src,
            edge_feature=tensor.edge_feature if self.use_edge_features else None,
            mode=self.degree_normalization)

        # (n_edges, ndim, 1) x (n_edges, 1, edim) -> (n_edges, ndim, edim)
        node_feature = propagate_node_features(
            node_feature=node_feature,
            edge_dst=tensor.edge_dst,
            edge_src=tensor.edge_src,
            edge_weight=edge_weight)

        if hasattr(self, 'kernel_decomp'):
            # (n_dim, unit, n_bases) x (e_dim, n_bases) -> (n_dim, unit, e_dim)
            kernel = tf.einsum(
                'ikb,eb->ike', self.kernel, self.kernel_decomp)
        else:
            kernel = self.kernel

        # (n_nodes, n_dim, e_dim) x (n_dim, unit, e_dim) -> (n_nodes, unit, e_dim)
        node_feature = tf.einsum('ij...,jk...->ik...', node_feature, kernel)

        if self.use_bias:
            node_feature += self.bias

        if self.use_edge_features:
            node_feature = tf.reduce_sum(node_feature, -1)

        if self.apply_self_projection:
            node_feature += self.self_projection(tensor.node_feature)

        return tensor.update({'node_feature': node_feature})

    def get_config(self):
        base_config = super().get_config()
        config = {
            'use_edge_features': self.use_edge_features,
            'self_projection': self.apply_self_projection,
            'degree_normalization': self.degree_normalization,
            'num_bases': self.num_bases,
            'use_bias': self.use_bias,
        }
        base_config.update(config)
        return base_config
