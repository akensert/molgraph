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
class GCNIIConv(gnn_layer.GNNLayer):

    '''Graph convolutional 'via Initial residual and Identity mapping' layer (GCNII).

    Implementation is based on Chen et al. (2020) [#]_.

    Example usage:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     sizes=[2, 3],
    ...     node_feature=[[1., 0.], [1., 0.], [1., 0.], [1., 0.], [0., 1.]],
    ...     edge_src=[1, 0, 3, 4, 2, 4, 3, 2],
    ...     edge_dst=[0, 1, 2, 2, 3, 3, 4, 4],
    ... )
    >>> gnn_model = tf.keras.Sequential([
    ...     molgraph.layers.GCNIIConv(units=16),
    ...     molgraph.layers.GCNIIConv(units=16),
    ...     molgraph.layers.GCNIIConv(units=16),
    ...     molgraph.layers.Readout(),
    ... ])
    >>> gnn_model(graph_tensor).shape
    TensorShape([2, 16])

    Args:
        units (int, None):
            Number of output units.
        alpha (float):
            Decides how much information of the initial node state (the
            original node features) should be passed to the next layer (alpha)
            vs. how much new information should be passed to the subsequent
            layer (1 - alpha). Takes a value between 0.0 and 1.0. In the original
            paper, alpha was set between 0.1 and 0.5, depending on the dataset.
        beta (float):
            Decides to what extent the kernel (projection) should be ignored.
            Takes a value between 0.0 and 1.0; a value set to 0.0 means that the
            kernel is ignored, a value set to 1.0 means that the kernel is fully
            applied. In the original paper, beta is set to log(lambda/l+1);
            l denotes the l:th layer, and lambda is a hyperparameter, set,
            in the original paper, between 0.5 and 1.5 depending on the dataset.
        variant (bool):
            Whether the GCNII variant should be used. Default to False.
        degree_normalization (str, None):
            The strategy for computing edge weights from degrees. Either of
            'symmetric', 'row' or None. If None, 'row' is used. Default to 'symmetric'.
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


    References:
        .. [#] https://arxiv.org/pdf/2007.02133v1.pdf

    '''

    def __init__(
        self,
        units: Optional[int] = None,
        alpha: float = 0.5,
        beta: float = 0.5,
        variant: bool = False,
        degree_normalization: str = 'symmetric',
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
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            use_edge_features=kwargs.pop('use_edge_features', False),
            **kwargs)

        self.alpha = alpha
        self.beta = beta
        self.variant = variant
        self.degree_normalization = degree_normalization
        self.apply_self_projection = self_projection
        self.residual = residual

    def _build(self, graph_tensor_spec: GraphTensor.Spec) -> None:

        self.projection = self.get_dense(self.units)

        node_dim = graph_tensor_spec.node_feature.shape[-1]
        
        if self.units != node_dim and not hasattr(self, 'node_resample'):
            self.node_resample = self.get_dense(self.units)

        if self.apply_self_projection:
            self.self_projection = self.get_dense(self.units)

    def _call(self, tensor: GraphTensor) -> GraphTensor:

        if hasattr(self, 'node_resample'):
            tensor = tensor.update({
                'node_feature': self.node_resample(tensor.node_feature)})

        if not hasattr(tensor, 'node_feature_initial'):
            tensor = tensor.update({
                'node_feature_initial': tensor.node_feature})

        edge_weight = gnn_ops.compute_edge_weights_from_degrees(
            edge_src=tensor.edge_src,
            edge_dst=tensor.edge_dst,
            edge_feature=None,
            mode=self.degree_normalization)

        tensor = tensor.update({'edge_weight': edge_weight})
        
        node_feature = tensor.propagate().node_feature

        identity = (
            (1 - self.alpha) * node_feature +
            self.alpha * tensor.node_feature_initial
        )

        if self.variant:
            node_feature = tf.concat([
                node_feature * (1 - self.alpha),
                tensor.node_feature_initial * self.alpha
            ], axis=1)
        else:
            node_feature = identity

        node_feature = (
            self.beta * self.projection(node_feature) +
            (1 - self.beta) * identity
        )

        if self.apply_self_projection:
            node_feature += self.self_projection(tensor.node_feature)

        return tensor.update({'node_feature': node_feature})

    def get_config(self):
        base_config = super().get_config()
        config = {
            'self_projection': self.apply_self_projection,
            'degree_normalization': self.degree_normalization,
            'alpha': self.alpha,
            'beta': self.beta,
            'variant': self.variant
        }
        base_config.update(config)
        return base_config
