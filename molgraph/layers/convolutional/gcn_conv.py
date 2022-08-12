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
from molgraph.layers.base import _BaseLayer
from molgraph.layers.ops import compute_edge_weights_from_degrees
from molgraph.layers.ops import propagate_node_features



@keras.utils.register_keras_serializable(package='molgraph')
class GCNConv(_BaseLayer):

    """Graph convolutional layer based on Kipf et al. [#]_ and
    Dwivedi et al. [#]_ (GCN); and Schlichtkrull et al. [#]_ (RGCN).

    References:

    .. [#] Kipf et al. https://arxiv.org/pdf/1609.02907.pdf
    .. [#] Dwivedi et al. https://arxiv.org/pdf/2003.00982.pdf
    .. [#] Schlichtkrull et al. https://arxiv.org/pdf/1703.06103.pdf
    """

    def __init__(
        self,
        units: Optional[int] = None,
        use_edge_features: bool = False,
        weight_normalization = 'symmetric',
        num_bases=None,
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
        self.weight_normalization = weight_normalization
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
            tensor.edge_dst,
            tensor.edge_src,
            tensor.edge_feature if self.use_edge_features else None,
            self.weight_normalization)

        # (n_edges, ndim, 1) x (n_edges, 1, edim) -> (n_edges, ndim, edim)
        node_feature = propagate_node_features(
            node_feature, tensor.edge_dst, tensor.edge_src, edge_weight)

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
            'weight_normalization': self.weight_normalization,
            'num_bases': self.num_bases,
            'use_bias': self.use_bias,
        }
        base_config.update(config)
        return base_config
