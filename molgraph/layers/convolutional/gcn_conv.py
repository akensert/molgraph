import tensorflow as tf
from tensorflow import keras 

from keras import initializers
from keras import regularizers
from keras import constraints

from typing import Optional
from typing import Callable
from typing import Union

from molgraph.internal import register_keras_serializable 

from molgraph.tensors.graph_tensor import GraphTensor

from molgraph.layers import gnn_layer
from molgraph.layers import gnn_ops


@register_keras_serializable(package='molgraph')
class GCNConv(gnn_layer.GNNLayer):

    """Graph convolutional layer (GCN).

    Implementation is based on Kipf et al. (2017) [#]_, Dwivedi et al. (2022) [#]_,
    and, for RGCN, Schlichtkrull et al. (2017) [#]_.

    Example usage:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     sizes=[2, 3],
    ...     node_feature=[[1., 0.], [1., 0.], [1., 0.], [1., 0.], [0., 1.]],
    ...     edge_src=[1, 0, 3, 4, 2, 4, 3, 2],
    ...     edge_dst=[0, 1, 2, 2, 3, 3, 4, 4],
    ... )
    >>> gnn_model = tf.keras.Sequential([
    ...     molgraph.layers.GCNConv(units=16),
    ...     molgraph.layers.GCNConv(units=16),
    ...     molgraph.layers.GCNConv(units=16),
    ...     molgraph.layers.Readout(),
    ... ])
    >>> gnn_model(graph_tensor).shape
    TensorShape([2, 16])

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
                Only relevant if edge features exist. Default to False.


    References:
        .. [#] https://arxiv.org/pdf/1609.02907.pdf
        .. [#] https://arxiv.org/pdf/2003.00982.pdf
        .. [#] https://arxiv.org/pdf/1703.06103.pdf
    """

    def __init__(
        self,
        units: Optional[int] = None,
        degree_normalization: Optional[str] = 'symmetric',
        num_bases: Optional[int] = None,
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
            use_bias=False,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            use_edge_features=kwargs.pop('use_edge_features', False),
            **kwargs)

        self.degree_normalization = degree_normalization
        self.num_bases = (
            int(num_bases) if isinstance(num_bases, (int, float)) else 0)
        self.apply_self_projection = self_projection
        self.use_bias = use_bias

    def _build(self, graph_tensor_spec: GraphTensor.Spec) -> None:

        kernel_shape = [self.node_dim, self.units]
        bias_shape = [self.units]


        if self.use_edge_features:

            edge_dim = graph_tensor_spec.edge_feature.shape[-1]

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

    def _call(self, tensor: GraphTensor) -> GraphTensor:

        if self.use_edge_features:
            node_feature = tf.expand_dims(tensor.node_feature, axis=-1)
        else:
            node_feature = tensor.node_feature

        edge_weight = gnn_ops.compute_edge_weights_from_degrees(
            edge_src=tensor.edge_src,
            edge_dst=tensor.edge_dst,
            edge_feature=tensor.edge_feature if self.use_edge_features else None,
            mode=self.degree_normalization)

        # (n_edges, ndim, 1) x (n_edges, 1, edim) -> (n_edges, ndim, edim)
        tensor_update = tensor.update({
            'node_feature': node_feature, 'edge_weight': edge_weight})
        node_feature = tensor_update.propagate().node_feature

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
            'self_projection': self.apply_self_projection,
            'degree_normalization': self.degree_normalization,
            'num_bases': self.num_bases,
            'use_bias': self.use_bias,
        }
        base_config.update(config)
        return base_config
