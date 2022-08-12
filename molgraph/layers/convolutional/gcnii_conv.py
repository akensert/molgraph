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
class GCNIIConv(_BaseLayer):

    """Graph convolutional \`via Initial residual and Identity mapping\` layer
    based on Chen et al. [#]_.

    Note
    ----
    alpha: float
        Decides how much information of the initial node state (the
        original node features) should be passed to the subsequent layers
        (alpha) vs. how much new information should be passed to the subsequent
        layers (1 - alpha). Takes a value between 0.0 and 1.0. In the original
        paper, alpha was set between 0.1 and 0.5, depending on the dataset.
    beta: float
        Decides to what extent the kernel (projection) should be ignored.
        A value set to 0.0 means that the kernel is ignored, a value set to 1.0
        means that the kernel is fully applied. Takes a value between 0.0 and
        1.0. In the original paper, beta is set to log(lambda/l+1)
        where lambda is a hyperparameter, set, in the original paper
        between 0.5 and 1.5 depending on the dataset; l denotes the l:th layer.
    variant: str, optional
        Whether the GCNII variant should be used.

    References
    ----------
    .. [#] Chen et al. https://arxiv.org/pdf/2007.02133v1.pdf

    """

    def __init__(
        self,
        units: Optional[int] = None,
        alpha: float = 0.5,
        beta: float = 0.5,
        variant: bool = False,
        weight_normalization: str = 'symmetric',
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

        self.alpha = alpha
        self.beta = beta
        self.variant = variant
        self.weight_normalization = weight_normalization
        self.apply_self_projection = self_projection
        self.residual = residual

    def subclass_build(
        self,
        node_feature_shape: tf.TensorShape,
        edge_feature_shape: Optional[tf.TensorShape] = None
    ) -> None:

        self.projection = self.get_dense(self.units)

        if (
            self.units != node_feature_shape[-1] and
            not hasattr(self, 'node_resample')
        ):
            self.node_resample = self.get_dense(self.units)

        if self.apply_self_projection:
            self.self_projection = self.get_dense(self.units)

    def subclass_call(self, tensor: GraphTensor) -> None:

        if hasattr(self, 'node_resample'):
            tensor = tensor.update({
                'node_feature': self.node_resample(tensor.node_feature)})

        if not hasattr(tensor, 'node_feature_initial'):
            tensor = tensor.update({'node_feature_initial': tensor.node_feature})

        edge_weight = compute_edge_weights_from_degrees(
            tensor.edge_dst, tensor.edge_src, None, self.weight_normalization)

        node_feature = propagate_node_features(
            tensor.node_feature, tensor.edge_dst, tensor.edge_src, edge_weight)

        identity = (
            (1 - self.alpha) * node_feature +
            self.alpha * tensor.node_feature_initial
        )

        if self.variant:
            node_feature = tf.concat([
                node_feature * (1 - self.alpha),
                tensor.node_features_initial * self.alpha
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
            'weight_normalization': self.weight_normalization,
            'alpha': self.alpha,
            'beta': self.beta,
            'variant': self.variant
        }
        base_config.update(config)
        return base_config
