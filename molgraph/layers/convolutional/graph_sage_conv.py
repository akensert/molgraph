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
from molgraph.layers.ops import compute_edge_weights_from_degrees
from molgraph.layers.ops import propagate_node_features



@keras.utils.register_keras_serializable(package='molgraph')
class GraphSageConv(BaseLayer):

    '''Graph sage convolution layer (GraphSage)

    Implementation is based on Hamilton et al. (2018) [#]_ and
    Dwivedi et al. (2022) [#]_.

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
    >>> # Build a model with GraphSageConv
    >>> gnn_model = tf.keras.Sequential([
    ...     tf.keras.Input(type_spec=graph_tensor.unspecific_spec),
    ...     molgraph.layers.GraphSageConv(16, activation='relu'),
    ...     molgraph.layers.GraphSageConv(16, activation='relu')
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
    >>> # Build a model with GraphSageConv
    >>> gnn_model = tf.keras.Sequential([
    ...     tf.keras.Input(type_spec=graph_tensor.unspecific_spec),
    ...     molgraph.layers.GraphSageConv(16, activation='relu'),
    ...     molgraph.layers.GraphSageConv(16, activation='relu')
    ... ])
    >>> gnn_model.output_shape
    (None, 16)

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
        .. [#] https://arxiv.org/pdf/1706.02216.pdf
        .. [#] https://arxiv.org/pdf/2003.00982.pdf

    '''

    def __init__(
        self,
        units: Optional[int] = None,
        aggregation_mode: str = 'mean',
        normalize: bool = True,
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

        self.aggregation_mode = aggregation_mode
        self.normalize = normalize if not batch_norm else False
        self.apply_self_projection = self_projection
        self.activation = activations.get('relu')

    def subclass_build(
        self,
        node_feature_shape: Optional[tf.TensorShape] = None,
        edge_feature_shape: Optional[tf.TensorShape] = None
    ) -> None:

        if self.aggregation_mode == 'max':
            self.node_src_projection = self.get_dense(self.units)
        elif self.aggregation_mode == 'lstm':
            self.lstm = layers.LSTM(self.units)

        if self.apply_self_projection:
            self.self_projection = self.get_dense(self.units)

        self.node_projection = self.get_dense(self.units)

    def subclass_call(self, tensor: GraphTensor) -> GraphTensor:

        if self.aggregation_mode == 'max':
            node_feature = self.node_src_projection(tensor.node_feature)
            node_feature = self.activation(node_feature)
            node_feature = propagate_node_features(
                node_feature=node_feature,
                edge_dst=tensor.edge_dst,
                edge_src=tensor.edge_src,
                mode=self.aggregation_mode)
        elif self.aggregation_mode == 'lstm':
            node_feature = self.lstm_aggregate(
                tensor.node_feature, tensor.edge_dst, tensor.edge_src)
        else:
            node_feature = propagate_node_features(
                node_feature=tensor.node_feature,
                edge_dst=tensor.edge_dst,
                edge_src=tensor.edge_src,
                mode=self.aggregation_mode)

        node_feature = tf.concat([node_feature, tensor.node_feature], axis=-1)

        node_feature = self.node_projection(node_feature)

        if self.apply_self_projection:
            node_feature += self.self_projection(tensor.node_feature)

        if self.normalize:
            node_feature = self.activation(node_feature)
            node_feature = tf.math.l2_normalize(node_feature, axis=1)

        return tensor.update({'node_feature': node_feature})

    def lstm_aggregate(self, node_feature, edge_dst, edge_src):

        def true_fn(node_feature, edge_dst, edge_src):
            """If edges exist, call this function"""

            node_indices = tf.unique(edge_dst)[0]
            num_nodes = tf.shape(node_feature)[0]
            #print(node_indices)

            # A bit of a hack to shuffle neighbor (source) nodes of the
            # destination nodes.
            random_indices = tf.random.shuffle(tf.range(tf.shape(edge_dst)[0]))
            edge_dst = tf.gather(edge_dst, random_indices)
            edge_src = tf.gather(edge_src, random_indices)
            sorted_indices = tf.argsort(edge_dst)
            edge_dst = tf.gather(edge_dst, sorted_indices)
            edge_dst -= tf.reduce_min(edge_dst)
            edge_src = tf.gather(edge_src, sorted_indices)

            # Gather source nodes followed by a partitioning of destination nodes
            node_feature = tf.RaggedTensor.from_value_rowids(
                tf.gather(node_feature, edge_src), edge_dst)
            node_feature = node_feature.to_tensor()
            # Pass to lstm for update
            node_feature = self.lstm(node_feature)

            node_feature_dim = tf.shape(node_feature)[-1]

            return tf.scatter_nd(
                indices=tf.expand_dims(node_indices, axis=-1),
                updates=node_feature,
                shape=(num_nodes, node_feature_dim))

        def false_fn(node_feature):
            """If no edges exist, call this function"""
            return tf.zeros(
                shape=(tf.shape(node_feature)[0], self.units),
                dtype=node_feature.dtype)

        return tf.cond(
            tf.greater(tf.shape(edge_dst)[0], 0),
            lambda: true_fn(node_feature, edge_dst, edge_src),
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
