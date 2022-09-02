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
from molgraph.layers.base import BaseLayer
from molgraph.layers.ops import propagate_node_features
from molgraph.layers.geometric import _radial_basis



@keras.utils.register_keras_serializable(package='molgraph')
class DTNNConv(BaseLayer):

    """Deep Tensor Neural Network (DTNN).

    Implementation is based on Schütt et al. (2017a) [#]_.

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
    ...         # edge_feature encodes distances between edge_dst and edge_src
    ...         'edge_feature': [[0.3, 0.3], [0.1, 0.2, 0.1, 0.4, 0.4, 0.2]],
    ...     }
    ... )
    >>> # Build a model with DTNNConv
    >>> gnn_model = tf.keras.Sequential([
    ...     tf.keras.Input(type_spec=graph_tensor.unspecific_spec),
    ...     molgraph.layers.DTNNConv(16, activation='relu'),
    ...     molgraph.layers.DTNNConv(16, activation='relu')
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
    ...         # edge_feature encodes distances between edge_dst and edge_src
    ...         'edge_feature': [0.3, 0.3, 0.1, 0.2, 0.1, 0.4, 0.4, 0.2],
    ...     }
    ... )
    >>> # Build a model with DTNNConv
    >>> gnn_model = tf.keras.Sequential([
    ...     tf.keras.Input(type_spec=graph_tensor.unspecific_spec),
    ...     molgraph.layers.DTNNConv(16, activation='relu'),
    ...     molgraph.layers.DTNNConv(16, activation='relu')
    ... ])
    >>> gnn_model.output_shape
    (None, 16)

    Args:
        units (int, None):
            The number of output units.
        distance_min (float):
            The smallest center (mean) to be used for the radial basis function.
            I.e., it defines the minimum distance between atom pairs; or the
            minimum electrostatic interaction between nuclei, in the case of
            Coulomb values. Default to -1.0.
        distance_max (float):
            The largest center (mean) to be used for the radial basis function.
            I.e., it defines the maximum distance between atom pairs; or the
            maximum electrostatic interaction between nuclei, in the case of
            Coulomb values. Default to 18.0.
        distance_granularity (float):
            The distance between each center (mean) of the radial basis function.
            The smaller the granularity, the more centers will be used.
            Default to 0.1.
        rbf_stddev (float, str):
            The standard deviation of the radial basis function. If 'auto',
            'distance_granularity' will be used as standard deviation.
            Default to 'auto'.
        self_projection (bool):
            Whether to apply self projection. Default to True.
        batch_norm: (bool):
            Whether to apply batch normalization to the output. Default to True.
        residual: (bool)
            Whether to add skip connection to the output. Default to True.
        dropout: (float, None):
            Dropout applied to the output of the layer. Default to None.
        activation (tf.keras.activations.Activation, callable, str, None):
            Activation function applied to the output of the layer. Default to None.
        use_bias (bool):
            Whether the layer should use biases. Default to True.
        kernel_initializer (tf.keras.initializers.Initializer, str):
            Initializer function for the kernels. Default to
            tf.keras.initializers.GlorotUniform().
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
        .. [#] https://www.nature.com/articles/ncomms13890
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
