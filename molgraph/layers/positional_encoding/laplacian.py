import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import activations

from typing import Union
from typing import Callable
from typing import Optional

from molgraph.internal import register_keras_serializable 

from molgraph.tensors.graph_tensor import GraphTensor


@register_keras_serializable(package='molgraph')
class LaplacianPositionalEncoding(layers.Layer):

    '''Laplacian positional encoding.

    Implementation based on Dwivedi et al. (2021) [#]_ and Belkin et al. (2003) [#]_.

    Example usage:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     sizes=[2, 3],
    ...     node_feature=[[1., 0.], [1., 0.], [1., 0.], [1., 0.], [0., 1.]],
    ...     edge_src=[1, 0, 3, 4, 2, 4, 3, 2],
    ...     edge_dst=[0, 1, 2, 2, 3, 3, 4, 4],
    ... )
    >>> model = tf.keras.Sequential([
    ...     molgraph.layers.LaplacianPositionalEncoding(16),
    ... ])
    >>> graph_tensor = model(graph_tensor)
    >>> graph_tensor.node_feature != 1.0
    <tf.Tensor: shape=(5, 2), dtype=bool, numpy=
    array([[ True,  True],
           [ True,  True],
           [ True,  True],
           [ True,  True],
           [ True,  True]])>

    Args:
        dim (int):
            The dimension of the positional encoding. Default to 8.
        activation (tf.keras.activations.Activation, callable, str, None):
            Activation function applied to the output of the layer. Default to None.
        use_bias (bool):
            Whether the layer should use biases. Default to False.
        kernel_initializer (tf.keras.initializers.Initializer, str):
            Initializer function for the kernel. Default to
            tf.keras.initializers.TruncatedNormal(stddev=0.005).
        bias_initializer (tf.keras.initializers.Initializer, str):
            Initializer function for the bias. Default to
            tf.keras.initializers.Constant(0.).
        kernel_regularizer (tf.keras.regularizers.Regularizer, None):
            Regularizer function applied to the kernel. Default to None.
        bias_regularizer (tf.keras.regularizers.Regularizer, None):
            Regularizer function applied to the bias. Default to None.
        activity_regularizer (tf.keras.regularizers.Regularizer, None):
            Regularizer function applied to the final output of the layer.
            Default to None.
        kernel_constraint (tf.keras.constraints.Constraint, None):
            Constraint function applied to the kernel. Default to None.
        bias_constraint (tf.keras.constraints.Constraint, None):
            Constraint function applied to the bias. Default to None.

    References:
        .. [#] https://arxiv.org/pdf/2012.09699.pdf
        .. [#] https://ieeexplore.ieee.org/document/6789755
    '''

    def __init__(
        self,
        dim: int = 8,
        activation: Union[None, str, Callable[[tf.Tensor], tf.Tensor]] = None,
        use_bias: bool = False,
        kernel_initializer: Union[
            str, initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_regularizer: Optional[regularizers.Regularizer] = None,
        activity_regularizer: Optional[regularizers.Regularizer] = None,
        kernel_constraint: Optional[constraints.Constraint] = None,
        bias_constraint: Optional[constraints.Constraint] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self._target_dim = dim
        self._positional_encoding_precomputed = False
        self._node_feature_shape = None
        self._positional_encoding_shape = None
        self._built = False

    def _build(
        self,
        node_feature: Union[tf.Tensor, tf.TensorShape],
        positional_encoding: Optional[Union[tf.Tensor, tf.TensorShape]] = None
    ) -> None:
        'Custom build method for building the layer.'
        self._built = True

        if hasattr(node_feature, "shape"):
            self._node_feature_shape = tf.TensorShape(node_feature.shape)
        else:
            self._node_feature_shape = tf.TensorShape(node_feature)

        if positional_encoding is not None:

            if hasattr(positional_encoding, "shape"):
                self._positional_encoding_shape = tf.TensorShape(
                    positional_encoding.shape)
            else:
                self._positional_encoding_shape = tf.TensorShape(
                    positional_encoding)

            self._positional_encoding_precomputed = True
            self._target_dim = self._positional_encoding_shape[-1]

        self.projection = layers.Dense(
            units=self._node_feature_shape[-1],
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint)

    def call(self, tensor: GraphTensor, training: Optional[bool] = None) -> GraphTensor:
        '''Defines the computation from inputs to outputs.

        This method should not be called directly, but indirectly
        via ``__call__()``. Upon first call, the layer is automatically
        built via ``_build()``.

        Args:
            tensor (GraphTensor):
                A graph tensor which serves as input to the layer.

        Returns:
            GraphTensor:
                A graph tensor with updated node features.
        '''
        tensor_orig = tensor
        if isinstance(tensor.node_feature, tf.RaggedTensor):
            tensor = tensor.merge()

        if not self._built:
            self._build(
                getattr(tensor, 'node_feature', None),
                getattr(tensor, 'node_position', None)
            )

        def random_sign_flip(positional_encoding):
            random_vals = tf.random.uniform((self._target_dim,))
            return tf.where(
                random_vals < 0.5, positional_encoding, -positional_encoding)

        if not self._positional_encoding_precomputed:
            positional_encoding = compute_positional_encoding(
                tensor, self._target_dim)
        else:
            positional_encoding = tensor.node_position

        if training:
            positional_encoding = random_sign_flip(positional_encoding)
        node_feature = tensor.node_feature + self.projection(positional_encoding)
        return tensor_orig.update({'node_feature': node_feature})

    @classmethod
    def from_config(cls, config):
        node_feature_shape = config.pop("node_feature_shape")
        positional_encoding_shape = config.pop("positional_encoding_shape")
        layer = cls(**config)
        if node_feature_shape is None:
            pass
        else:
            layer._build(
                node_feature_shape, positional_encoding_shape
            )
        return layer

    def get_config(self):
        base_config = super().get_config()
        config = {
            'dim': self._target_dim,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'node_feature_shape': self._node_feature_shape,
            'positional_encoding_shape': self._positional_encoding_shape,
        }
        base_config.update(config)
        return base_config


def compute_normalized_laplacian(adjacency, num_nodes):
    degree = tf.math.bincount(
        tf.cast(adjacency[:, 1], tf.int32), dtype=adjacency.dtype)
    degree = tf.gather(degree, adjacency)
    adjacency_norm = tf.reduce_prod(tf.cast(degree, tf.float32), 1) ** -0.5
    laplacian_norm = tf.scatter_nd(
        adjacency, -adjacency_norm, (num_nodes, num_nodes))
    laplacian_norm += tf.eye(num_nodes, dtype=tf.float32)
    return laplacian_norm

def compute_eigen_vectors(laplacian, target_dim):
    _, eig_vec = tf.linalg.eig(laplacian)
    eig_vec = tf.math.real(eig_vec)
    pos_enc = eig_vec[:, 1: target_dim + 1]
    dim = tf.shape(pos_enc)[1]
    if dim < target_dim:
        pos_enc = tf.pad(pos_enc, [(0, 0), (0, target_dim - dim)])
    return pos_enc

def compute_positional_encoding(tensor, target_dim):

    num_nodes = tf.shape(tensor.node_feature)[0]
    adjacency = tf.stack([tensor.edge_src, tensor.edge_dst], axis=1)

    laplacian_norm = compute_normalized_laplacian(adjacency, num_nodes)

    graph_indicator = tensor.graph_indicator

    positional_encodings = tf.TensorArray(
        tf.float32, size=0, dynamic_size=True, infer_shape=False,
        element_shape=tf.TensorShape((None, target_dim)))

    for i in tf.range(tf.reduce_max(graph_indicator) + 1):

        indices = tf.where(graph_indicator == i)[:, 0]

        sliced_laplacian = tf.gather(tf.gather(
            laplacian_norm, indices, axis=0), indices, axis=1)

        positional_encoding = compute_eigen_vectors(
            sliced_laplacian, target_dim)

        positional_encodings = positional_encodings.write(
            positional_encodings.size(), positional_encoding)

    return positional_encodings.concat()
