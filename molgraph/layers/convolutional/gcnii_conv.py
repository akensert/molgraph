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
from molgraph.layers.base import BaseLayer
from molgraph.layers.ops import compute_edge_weights_from_degrees
from molgraph.layers.ops import propagate_node_features



@keras.utils.register_keras_serializable(package='molgraph')
class GCNIIConv(BaseLayer):

    '''Graph convolutional 'via Initial residual and Identity mapping' layer (GCNII).

    Implementation is based on Chen et al. (2020) [#]_.

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
    >>> # Build a model with GCNIIConv
    >>> gnn_model = tf.keras.Sequential([
    ...     tf.keras.Input(type_spec=graph_tensor.unspecific_spec),
    ...     molgraph.layers.GCNIIConv(16, activation='relu'),
    ...     molgraph.layers.GCNIIConv(16, activation='relu')
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
    >>> # Build a model with GCNIIConv
    >>> gnn_model = tf.keras.Sequential([
    ...     tf.keras.Input(type_spec=graph_tensor.unspecific_spec),
    ...     molgraph.layers.GCNIIConv(16, activation='relu'),
    ...     molgraph.layers.GCNIIConv(16, activation='relu')
    ... ])
    >>> gnn_model.output_shape
    (None, 16)

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
        self.degree_normalization = degree_normalization
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
            edge_dst=tensor.edge_dst,
            edge_src=tensor.edge_src,
            edge_feature=None,
            mode=self.degree_normalization)

        node_feature = propagate_node_features(
            node_feature=tensor.node_feature,
            edge_dst=tensor.edge_dst,
            edge_src=tensor.edge_src,
            edge_weight=edge_weight)

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
            'degree_normalization': self.degree_normalization,
            'alpha': self.alpha,
            'beta': self.beta,
            'variant': self.variant
        }
        base_config.update(config)
        return base_config
