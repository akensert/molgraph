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

from molgraph.layers.geometric import radial_basis


@register_keras_serializable(package='molgraph')
class DTNNConv(gnn_layer.GNNLayer):

    """Deep Tensor Neural Network (DTNN).

    Implementation is based on Schütt et al. (2017a) [#]_.

    Example usage:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     sizes=[2, 3],
    ...     node_feature=[[1., 0.], [1., 0.], [1., 0.], [1., 0.], [0., 1.]],
    ...     edge_src=[1, 0, 3, 4, 2, 4, 3, 2],
    ...     edge_dst=[0, 1, 2, 2, 3, 3, 4, 4],
    ...     # edge_feature encodes e.g. distances between edge_src and edge_dst
    ...     edge_feature=[0.3, 0.3, 0.1, 0.2, 0.1, 0.4, 0.4, 0.2],
    ... )
    >>> gnn_model = tf.keras.Sequential([
    ...     molgraph.layers.DTNNConv(units=16),
    ...     molgraph.layers.DTNNConv(units=16),
    ...     molgraph.layers.DTNNConv(units=16),
    ...     molgraph.layers.Readout(),
    ... ])
    >>> gnn_model(graph_tensor).shape
    TensorShape([2, 16])

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
        normalization: (None, str, bool):
            Whether to apply layer normalization to the output. If batch 
            normalization is desired, pass 'batch_norm'. Default to None.
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
        **kwargs: Valid (optional) keyword arguments are:

            *   `name` (str): Name of the layer instance.
            *   `update_step` (tf.keras.layers.Layer): Applies post-processing 
                step on the output (produced by `_call`). If passed, 
                `normalization`, `residual`, `activation` and `dropout` 
                parameters will be ignored. If None, a default post-processing 
                step will be used (taking into consideration the aforementioned 
                parameters). Default to None.
                
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
        normalization: Union[None, str, bool] = None,
        residual: bool = True,
        dropout: Optional[float] = None,
        activation: Union[None, str, Callable[[tf.Tensor], tf.Tensor]] = None,
        use_bias: bool = True,
        kernel_initializer: Union[
            str, initializers.Initializer, None] = 'glorot_uniform',
        bias_initializer: Union[
            str, initializers.Initializer, None] = 'zeros',
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
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            use_edge_features=kwargs.pop('use_edge_features', True),
            **kwargs)

        self.apply_self_projection = self_projection
        self.distance_min = distance_min
        self.distance_max = distance_max
        self.distance_granularity = distance_granularity
        self.rbf_stddev = rbf_stddev

        self.rbf = radial_basis.RadialBasis(
            self.distance_min,
            self.distance_max,
            self.distance_granularity,
            self.rbf_stddev
        )

    def _build(self, graph_tensor_spec: GraphTensor.Spec) -> None:
        
        self.edge_projection = self.get_dense(self.units)
        self.node_projection_1 = self.get_dense(self.units)
        self.node_projection_2 = self.get_dense(self.units)
        self.node_projection_3 = self.get_dense(self.units)
        if self.apply_self_projection:
            self.self_projection = self.get_dense(self.units)

    def _call(self, tensor: GraphTensor) -> GraphTensor:

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
