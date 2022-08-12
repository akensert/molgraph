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
class GINConv(_BaseLayer):

    """Graph isomorphism convolution layer based on Dwivedi et al. [#]_
    and Xu et al. [#]_.

    References
    ----------
    .. [#] Dwivedi et al. https://arxiv.org/pdf/2003.00982.pdf
    .. [#] Xu et al. https://arxiv.org/pdf/1810.00826.pdf

    """

    def __init__(
        self,
        units: Optional[int] = None,
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

        self.batch_norm = batch_norm
        self.activation = activations.get('relu')
        self.apply_self_projection = self_projection

    def subclass_build(
        self,
        node_feature_shape: Optional[tf.TensorShape] = None,
        edge_feature_shape: Optional[tf.TensorShape] = None
    ) -> None:

        self.projection_1 = self.get_dense(self.units)
        self.projection_2 = self.get_dense(self.units)
        self.epsilon = self.add_weight(
            shape=(),
            initializer='zeros',
            regularizer=None,
            trainable=True,
            name='epsilon')

        if self.batch_norm:
            self.batch_norm = layers.BatchNormalization()
        else:
            self.batch_norm = None

        if self.apply_self_projection:
           self.self_projection = self.get_dense(self.units)

    def subclass_call(self, tensor: GraphTensor) -> GraphTensor:

        node_feature = propagate_node_features(
            tensor.node_feature, tensor.edge_dst, tensor.edge_src)

        node_feature = (
            (1 + self.epsilon) * tensor.node_feature + node_feature)

        node_feature = self.projection_1(node_feature)

        if self.apply_self_projection:
            node_feature += self.self_projection(tensor.node_feature)

        if self.batch_norm is not None:
            node_feature = self.batch_norm(node_feature)

        node_feature = self.activation(node_feature)

        node_feature = self.projection_2(node_feature)

        return tensor.update({'node_feature': node_feature})

    def get_config(self):
        base_config = super().get_config()
        config = {
            'self_projection': self.apply_self_projection,
        }
        base_config.update(config)
        return base_config
