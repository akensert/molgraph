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
from molgraph.layers.ops import softmax_edge_weights
from molgraph.layers.ops import propagate_node_features
from molgraph.layers.ops import reduce_features



@keras.utils.register_keras_serializable(package='molgraph')
class GraphTransformerConv(_BaseLayer):

    """Graph transformer layer based on Dwivedi et al. [#]_.

    References:
    
    .. [#] Dwivedi et al. https://arxiv.org/pdf/2012.09699.pdf

    """

    def __init__(
        self,
        units: Optional[int] = 128,
        use_edge_features: bool = True,
        num_heads: int = 8,
        merge_mode: Optional[str] = 'concat',
        self_projection: bool = True,
        norm_mode: Optional[str] = 'layer_norm',
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
        kwargs.pop("batch_norm", None)
        kwargs.pop("residual", None)
        kwargs.pop("dropout", None)
        kwargs.pop("activation", None)

        super().__init__(
            units=units,
            batch_norm=None,
            residual=None,
            dropout=None,
            activation=None,
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
        self.num_heads = num_heads
        self.merge_mode = merge_mode
        self.apply_self_projection = self_projection
        self.norm_mode = norm_mode
        self.residual = residual
        self.dropout = dropout
        self.activation = activation

        self.output_units = self.units

    def subclass_build(
        self,
        node_feature_shape: Optional[tf.TensorShape] = None,
        edge_feature_shape: Optional[tf.TensorShape] = None
    ) -> None:

        if self.units != self.node_dim and self.residual:
            self.node_resample = self.get_dense(self.units)

        self.use_edge_features = (
            self.use_edge_features and edge_feature_shape is not None
        )
        if self.use_edge_features:
            if self.units != self.edge_dim and self.residual:
                self.edge_resample = self.get_dense(self.units)

        if not self.output_units:
            self.output_units = self.units

        if self.merge_mode == 'concat':
            if not self.units or (self.units % self.num_heads != 0):
                raise ValueError(
                    "`merge_mode` was set to `concat` and hence " +
                    " need `units` to be divisble by `num_heads`")
            self.units //= self.num_heads

        self.query_projection = self.get_einsum_dense(
            'ij,jkh->ihk', (self.num_heads, self.units))

        self.key_projection = self.get_einsum_dense(
            'ij,jkh->ihk', (self.num_heads, self.units))

        self.value_projection = self.get_einsum_dense(
            'ij,jkh->ihk', (self.num_heads, self.units))

        if self.apply_self_projection:
            self.self_projection = self.get_einsum_dense(
                'ij,jkh->ihk', (self.num_heads, self.units))

        # feed forward network for node features
        self.node_projection_1 = self.get_dense(self.output_units)
        self.node_projection_2 = self.get_dense(self.output_units)
        self.node_projection_3 = self.get_dense(self.output_units)
        self.node_activation = activations.get(self.activation)

        if self.dropout:
            self.node_dropout_1 = layers.Dropout(self.dropout)
            self.node_dropout_2 = layers.Dropout(self.dropout)

        if self.norm_mode == 'batch_norm':
            self.node_normalization_1 = layers.BatchNormalization()
            self.node_normalization_2 = layers.BatchNormalization()
        elif self.norm_mode == 'layer_norm':
            self.node_normalization_1 = layers.LayerNormalization()
            self.node_normalization_2 = layers.LayerNormalization()

        if self.use_edge_features:

            self.edge_gate_projection = self.get_einsum_dense(
                'ij,jkh->ihk', (self.num_heads, self.units))

            # feed forward network for edge features
            if self.update_edge_features:
                self.edge_projection_1 = self.get_dense(self.output_units)
                self.edge_projection_2 = self.get_dense(self.output_units)
                self.edge_projection_3 = self.get_dense(self.output_units)
                self.edge_activation = activations.get(self.activation)

                if self.dropout:
                    self.edge_dropout_1 = layers.Dropout(self.dropout)
                    self.edge_dropout_2 = layers.Dropout(self.dropout)

                if self.norm_mode == 'batch_norm':
                    self.edge_normalization_1 = layers.BatchNormalization()
                    self.edge_normalization_2 = layers.BatchNormalization()
                elif self.norm_mode == 'layer_norm':
                    self.edge_normalization_1 = layers.LayerNormalization()
                    self.edge_normalization_2 = layers.LayerNormalization()

        if self.merge_mode == 'concat':
            self.units *= self.num_heads

    def subclass_call(self, tensor: GraphTensor) -> GraphTensor:

        value = self.value_projection(tensor.node_feature)

        # Edge dependent. In rare cases, we input a graph with a single
        # node and no edges; below would throw an error in such cases.

        # Apply linear transformation to node and edge features
        query = self.query_projection(tensor.node_feature)
        key = self.key_projection(tensor.node_feature)

        # Gather self nodes' queries and corresponding neighbor nodes' keys
        query = tf.gather(query, tensor.edge_dst)
        key = tf.gather(key, tensor.edge_src)
        # tf.gather(value, edge_src) will be run inside self.propagate_features(..)

        attention_score = query * key

        # Rescale the attention scores
        attention_score = attention_score / tf.math.sqrt(float(self.units))

        if self.use_edge_features:
            edge_gate = self.edge_gate_projection(tensor.edge_feature)
            attention_score *= edge_gate

            if self.update_edge_features:
                edge_feature = self.feed_forward_network(
                    'edge', attention_score, tensor.edge_feature)
                tensor = tensor.update({'edge_feature': edge_feature})

        attention_score = softmax_edge_weights(attention_score, tensor.edge_dst)

        node_feature = propagate_node_features(
            value, tensor.edge_dst, tensor.edge_src, attention_score)

        if self.apply_self_projection:
            node_feature += self.self_projection(tensor.node_feature)

        node_feature = self.feed_forward_network(
            'node', node_feature, tensor.node_feature)

        return tensor.update({'node_feature': node_feature})

    def feed_forward_network(self, feature_type, feature, feature_residual):

        feature = reduce_features(
            feature, self.merge_mode, self.units)

        if self.dropout:
            feature = getattr(self, f'{feature_type}_dropout_1')(feature)

        feature = getattr(self, f'{feature_type}_projection_1')(feature)

        if self.residual:
            if hasattr(self, f'{feature_type}_resample'):
                feature_residual = getattr(
                    self, f'{feature_type}_resample')(feature_residual)
            feature += feature_residual

        if self.norm_mode:
            feature = getattr(self, f'{feature_type}_normalization_1')(feature)

        feature_residual = feature

        feature = getattr(self, f'{feature_type}_projection_2')(feature)
        feature = getattr(self, f'{feature_type}_activation')(feature)

        if self.dropout:
            feature = getattr(self, f'{feature_type}_dropout_2')(feature)

        feature = getattr(self, f'{feature_type}_projection_3')(feature)

        if self.residual:
            feature += feature_residual

        if self.norm_mode:
            feature = getattr(self, f'{feature_type}_normalization_2')(feature)

        return feature

    def get_config(self):
        base_config = super().get_config()
        config = {
            'use_edge_features': self.use_edge_features,
            'num_heads': self.num_heads,
            'merge_mode': self.merge_mode,
            'self_projection': self.apply_self_projection,
            'norm_mode': self.norm_mode,
            'residual': self.residual,
            'dropout': self.dropout,
            'activation': activations.serialize(self.activation),
        }
        base_config.update(config)
        return base_config
