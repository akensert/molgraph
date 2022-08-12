import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.keras import activations

import numpy as np

from typing import Optional
from typing import Callable
from typing import Union
from typing import Tuple

from molgraph.tensors.graph_tensor import GraphTensor
from molgraph.layers.base import _BaseLayer
from molgraph.layers.ops import propagate_node_features
from molgraph.layers.geometric import _radial_basis


@keras.utils.register_keras_serializable(package='molgraph')
class DTNNConv(_BaseLayer):

    """Deep Tensor Neural Network (DTNN) based on Schütt et al. [#]_.

    References
    ----------
    .. [#] Schütt et al. (b) https://www.nature.com/articles/ncomms13890
    """


    def __init__(
        self,
        units: Optional[int] = None,
        distance_min: float = -1.0,
        distance_max: float = 18.0,
        distance_granularity: float = 0.1,
        rbf_stddev: Optional[Union[float, str]] = 'auto',
        self_projection: bool = True,
        batch_norm: bool = True,
        residual: bool = True,
        dropout: Optional[float] = None,
        activation: Union[None, str, Callable[[tf.Tensor], tf.Tensor]] = None,
        use_bias: bool = True,
        kernel_initializer: Union[
            str, initializers.Initializer
        ] = initializers.GlorotUniform(),
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

        self.apply_self_projection = self_projection
        self.distance_min = distance_min
        self.distance_max = distance_max
        self.distance_granularity = distance_granularity
        self.rbf_stddev = rbf_stddev

        self.rbf = _radial_basis.RadialBasis(
            self.distance_min,
            self.distance_max,
            self.distance_granularity,
            self.rbf_stddev
        )

    def subclass_build(
        self,
        node_feature_shape: Optional[tf.TensorShape] = None,
        edge_feature_shape: Optional[tf.TensorShape] = None
    ) -> None:
        self.edge_projection = self.get_dense(self.units)
        self.node_projection_1 = self.get_dense(self.units)
        self.node_projection_2 = self.get_dense(self.units)
        self.node_projection_3 = self.get_dense(self.units)
        if self.apply_self_projection:
            self.self_projection = self.get_dense(self.units)

    def subclass_call(self, tensor: GraphTensor) -> GraphTensor:

        num_segments = tf.shape(tensor.node_feature)[0]

        node_feature = self.node_projection_1(tensor.node_feature)
        node_feature = tf.gather(node_feature, tensor.edge_src)

        edge_weight = self.rbf(tensor.edge_feature)
        edge_weight = self.edge_projection(edge_weight)

        node_feature *= edge_weight

        node_feature = self.node_projection_2(node_feature)

        node_feature = tf.nn.tanh(node_feature)

        node_feature = tf.math.unsorted_segment_sum(
            node_feature, tensor.edge_dst, num_segments)

        if self.apply_self_projection:
            node_feature += self.self_projection(tensor.node_feature)

        return tensor.update({'node_feature': node_feature})

    def get_config(self):
        base_config = super().get_config()
        base_config.update({
            'self_projection': self.apply_self_projection,
            'distance_min': self.distance_min,
            'distance_max': self.distance_max,
            'distance_granularity': self.distance_granularity,
            'rbf_stddev': self.rbf_stddev
        })
        return base_config
