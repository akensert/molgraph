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
from molgraph.layers.ops import softmax_edge_weights
from molgraph.layers.ops import propagate_node_features



@keras.utils.register_keras_serializable(package='molgraph')
class GatedGCNConv(BaseLayer):

    '''Gated graph convolutional layer (GatedGCN).

    Implementation is based on Dwivedi et al. (2022) [#]_, Bresson et al. (2019) [#]_,
    Joshi et al. (2019) [#]_, and Bresson et al. (2018) [#]_.

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
    >>> # Build a model with GatedGCNConv
    >>> gnn_model = tf.keras.Sequential([
    ...     tf.keras.Input(type_spec=graph_tensor.unspecific_spec),
    ...     molgraph.layers.GatedGCNConv(16, activation='relu'),
    ...     molgraph.layers.GatedGCNConv(16, activation='relu')
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
    >>> # Build a model with GatedGCNConv
    >>> gnn_model = tf.keras.Sequential([
    ...     tf.keras.Input(type_spec=graph_tensor.unspecific_spec),
    ...     molgraph.layers.GatedGCNConv(16, activation='relu'),
    ...     molgraph.layers.GatedGCNConv(16, activation='relu')
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
        .. [#] https://arxiv.org/pdf/1906.03412.pdf
        .. [#] https://arxiv.org/pdf/1906.01227.pdf
        .. [#] https://arxiv.org/pdf/1711.07553.pdf

    '''

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
        kwargs['update_edge_features'] = (
            kwargs.get('update_edge_features', True) and use_edge_features
        )
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
        self.gate_activation = activations.get('sigmoid')
        self.apply_self_projection = self_projection

    def subclass_build(
        self,
        node_feature_shape: Optional[tf.TensorShape] = None,
        edge_feature_shape: Optional[tf.TensorShape] = None
    ) -> None:

        self.use_edge_features = (
            self.use_edge_features and edge_feature_shape is not None
        )
        if self.use_edge_features:
            self.edge_gate_projection = self.get_dense(self.units)

        self.node_src_gate_projection = self.get_dense(self.units)
        self.node_dst_gate_projection = self.get_dense(self.units)

        self.node_projection = self.get_dense(self.units)
        if self.apply_self_projection:
            self.self_projection = self.get_dense(self.units)

    def subclass_call(self, tensor: GraphTensor) -> GraphTensor:

        # Edge dependent (`tensor.edge_src is not None`), from here
        node_feature = self.node_projection(tensor.node_feature)

        node_feature_src = tf.gather(
            tensor.node_feature, tensor.edge_src)
        node_feature_dst = tf.gather(
            tensor.node_feature, tensor.edge_dst)

        gate = self.node_src_gate_projection(node_feature_src)
        gate += self.node_dst_gate_projection(node_feature_dst)

        if self.use_edge_features:
            edge_feature = self.edge_gate_projection(tensor.edge_feature)
            gate += edge_feature
            tensor = tensor.update({'edge_feature': gate})

        gate = self.gate_activation(gate)
        gate = softmax_edge_weights(
            edge_weight=gate,
            edge_dst=tensor.edge_dst,
            exponentiate=False)

        node_feature = propagate_node_features(
            node_feature=node_feature,
            edge_dst=tensor.edge_dst,
            edge_src=tensor.edge_src,
            edge_weight=gate)

        if self.apply_self_projection:
            node_feature += self.self_projection(tensor.node_feature)

        return tensor.update({'node_feature': node_feature})

    def get_config(self):
        base_config = super().get_config()
        config = {
            'use_edge_features': self.use_edge_features,
            'self_projection': self.apply_self_projection,
        }
        base_config.update(config)
        return base_config
