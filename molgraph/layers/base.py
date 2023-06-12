import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import activations
from keras.utils import tf_utils

from abc import ABC, abstractmethod

from typing import Optional
from typing import Callable
from typing import Union
from typing import List
from typing import Tuple
from typing import Type
from typing import TypeVar

from molgraph.tensors.graph_tensor import GraphTensor


Shape = Union[List[int], Tuple[int, ...], tf.TensorShape]
DType = Union[str, tf.DType]
Config = TypeVar('Config', bound=dict)
Activation = Optional[Union[Callable[[tf.Tensor], tf.Tensor], str]]


@keras.utils.register_keras_serializable(package='molgraph')
class BaseLayer(layers.Layer, ABC):

    'Base layer for the built-in GNN layers.'

    def __init__(
        self,
        units: Optional[int] = None,
        batch_norm: bool = False,
        residual: bool = False,
        dropout: Optional[float] = None,
        activation: Activation = None,
        use_bias: bool = False,
        kernel_initializer: Union[str, initializers.Initializer, None] = None,
        bias_initializer: Union[str, initializers.Initializer, None] = None,
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_regularizer: Optional[regularizers.Regularizer] = None,
        activity_regularizer: Optional[regularizers.Regularizer] = None,
        kernel_constraint: Optional[constraints.Constraint] = None,
        bias_constraint: Optional[constraints.Constraint] = None,
        **kwargs
    ) -> None:

        self.update_edge_features = kwargs.pop('update_edge_features', False)
        kwargs.pop('use_edge_features', False)

        layers.Layer.__init__(self, **kwargs)

        if kernel_initializer is None:
            kernel_initializer = initializers.TruncatedNormal(stddev=0.005)
        if bias_initializer is None:
            bias_initializer = initializers.Constant(0.)
    
        self.units = units
        self._batch_norm = batch_norm
        self._residual = residual
        self._dropout = dropout
        self._activation = activations.get(activation)
        self._use_bias = use_bias
        self._kernel_initializer = initializers.get(kernel_initializer)
        self._bias_initializer = initializers.get(bias_initializer)
        self._kernel_regularizer = regularizers.get(kernel_regularizer)
        self._bias_regularizer = regularizers.get(bias_regularizer)
        self._activity_regularizer = regularizers.get(activity_regularizer)
        self._kernel_constraint = constraints.get(kernel_constraint)
        self._bias_constraint = constraints.get(bias_constraint)
        self._built = False
        self._node_feature_shape, self._edge_feature_shape = None, None

    @abstractmethod
    def subclass_call(self, inputs: GraphTensor) -> GraphTensor:
        pass

    @abstractmethod
    def subclass_build(
        self,
        node_feature_shape: tf.TensorShape,
        edge_feature_shape: Optional[tf.TensorShape]
    ) -> None:
        pass

    def call(self, tensor: GraphTensor) -> GraphTensor:
        '''Defines the computation from inputs to outputs.

        This method should not be called directly, but indirectly
        via ``__call__()``. Upon first call, the layer is automatically
        built via ``_build()``.

        Args:
            tensor (GraphTensor):
                Input to the layer.

        Returns:
            GraphTensor:
                A ``GraphTensor`` with updated features. For some layers,
                both the ``node_features`` field and the ``edge_features``
                field (of the ``GraphTensor``) are updated.
        '''
        tensor_orig = tensor
        if isinstance(tensor.node_feature, tf.RaggedTensor):
            tensor = tensor.merge()

        if not self._built:
            self._build(
                getattr(tensor, 'node_feature', None),
                getattr(tensor, 'edge_feature', None)
            )

        tensor_update = self.subclass_call(tensor)
        tensor_update = self._process_output(tensor_update, tensor)

        return tensor_orig.update({
            k: v for (k, v) in tensor_update._data.items() if
            k not in ['edge_dst', 'edge_src', 'graph_indicator']
        })

    def _build(
        self,
        node_feature: Union[tf.Tensor, Shape],
        edge_feature: Optional[Union[tf.Tensor, Shape]] = None
    ) -> None:
        'Custom build method for building the layer.'
        self._built = True

        if hasattr(node_feature, "shape"):
            self._node_feature_shape = tf.TensorShape(node_feature.shape)
        else:
            self._node_feature_shape = tf.TensorShape(node_feature)

        if edge_feature is not None:
            if hasattr(edge_feature, 'shape'):
                self._edge_feature_shape = tf.TensorShape(edge_feature.shape)
            else:
                self._edge_feature_shape = tf.TensorShape(edge_feature)
        else:
            self._edge_feature_shape = None

        with tf_utils.maybe_init_scope(self):

            self.node_dim = self._node_feature_shape[-1]

            if not self.units:
                self.units = self.node_dim

            if self.units != self.node_dim and self._residual:
                # If we are going transform node features to a higher dimension
                # and subsequently want to perform a skip connection (residual),
                # we need to first upsample node features.
                self.node_resample = self.get_dense(self.units)

            self._node_batch_norm = (
                layers.BatchNormalization() if self._batch_norm else None)

            self._node_activation = activations.get(self._activation)

            self._node_dropout = (
                layers.Dropout(self._dropout) if self._dropout else None)

            self.update_edge_features = (
                self.update_edge_features and
                self._edge_feature_shape is not None)

            if self.update_edge_features:

                self.edge_dim = self._edge_feature_shape[-1]
                if self.units != self.edge_dim and self._residual:
                    self.edge_resample = self.get_dense(self.units)

                self._edge_batch_norm = (
                    layers.BatchNormalization() if self._batch_norm else None)

                self._edge_activation = activations.get(self._activation)

                self._edge_dropout = (
                    layers.Dropout(self._dropout) if self._dropout else None)

            self.subclass_build(
                self._node_feature_shape, self._edge_feature_shape)

    def _process_output(
        self,
        tensor_update: GraphTensor,
        tensor: GraphTensor
    ) -> GraphTensor:
        node_feature = tensor_update.node_feature
        if self._batch_norm:
            node_feature = self._node_batch_norm(node_feature)
        node_feature = self._node_activation(node_feature)
        if self._residual:
            if hasattr(self, 'node_resample'):
                node_feature_residual = self.node_resample(
                    tensor.node_feature)
            else:
                node_feature_residual = tensor.node_feature
            node_feature += node_feature_residual
        if self._dropout:
            node_feature = self._node_dropout(node_feature)
        tensor_update = tensor_update.update({'node_feature': node_feature})

        if self.update_edge_features:
            edge_feature = tensor_update.edge_feature
            if self._batch_norm:
                edge_feature = self._edge_batch_norm(edge_feature)
            edge_feature = self._edge_activation(edge_feature)
            if self._residual:
                if hasattr(self, 'edge_resample'):
                    edge_feature_residual = self.edge_resample(
                        tensor.edge_feature)
                else:
                    edge_feature_residual = tensor.edge_feature
                edge_feature += edge_feature_residual
            if self._dropout:
                edge_feature = self._edge_dropout(edge_feature)
            tensor_update = tensor_update.update({'edge_feature': edge_feature})

        return tensor_update

    def get_common_kwargs(self):
        common_kwargs = dict(
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
        )
        kernel_initializer = self._kernel_initializer.__class__.from_config(
            self._kernel_initializer.get_config()
        )
        bias_initializer = self._bias_initializer.__class__.from_config(
            self._bias_initializer.get_config()
        )
        common_kwargs["kernel_initializer"] = kernel_initializer
        common_kwargs["bias_initializer"] = bias_initializer
        return common_kwargs
    
    def get_kernel(
        self,
        shape: Shape,
        dtype: DType = tf.float32,
        name: str = 'kernel'
    ) -> tf.Variable:
        common_kwargs = self.get_common_kwargs()
        return self.add_weight(
            name=name,
            shape=shape,
            dtype=dtype,
            initializer=common_kwargs['kernel_initializer'],
            regularizer=common_kwargs['kernel_regularizer'],
            constraint=common_kwargs['kernel_constraint'],
            trainable=True
        )

    def get_bias(self,
        shape: Shape,
        dtype: DType = tf.float32,
        name: str = 'bias'
    ) -> tf.Variable:
        common_kwargs = self.get_common_kwargs()
        return self.add_weight(
            name=name,
            shape=shape,
            dtype=dtype,
            initializer=common_kwargs['bias_initializer'],
            regularizer=common_kwargs['bias_regularizer'],
            constraint=common_kwargs['bias_constraint'],
            trainable=True
        )

    def get_dense(
        self,
        units: int,
        activation: Activation = None,
    ) -> layers.Dense:
        return layers.Dense(
            units,
            activation=activation,
            use_bias=self._use_bias,
            **self.get_common_kwargs())

    def get_einsum_dense(
        self,
        equation: str,
        output_shape: Shape,
        activation: Activation = None,
    ) -> layers.EinsumDense:

        if self._use_bias:
            bias_axes = equation.split('->')[-1][1:]
            if len(bias_axes) == 0:
                bias_axes = None
        else:
            bias_axes = None

        return layers.EinsumDense(
            equation,
            output_shape,
            activation=activation,
            bias_axes=bias_axes,
            **self.get_common_kwargs())

    def compute_output_shape(
        self,
        input_shape: Shape
    ) -> tf.TensorShape:
        inner_dim = self.units
        if getattr(self, 'merge_mode', None) == 'concat':
            if hasattr(self, 'num_heads'):
                inner_dim *= self.num_heads
            elif hasattr(self, 'num_kernels'):
                inner_dim *= getattr(self, 'num_kernels', 1)
        return tf.TensorShape(input_shape[:-1]).concatenate([inner_dim])

    @classmethod
    def from_config(cls: Type['BaseLayer'], config: Config) -> 'BaseLayer':
        node_feature_shape = config.pop('node_feature_shape')
        edge_feature_shape = config.pop('edge_feature_shape')
        layer = cls(**config)
        if None in [node_feature_shape, edge_feature_shape]:
            pass
        else:
            layer._build(node_feature_shape, edge_feature_shape)
        return layer

    def get_config(self) -> Config:
        config = super().get_config()
        config.update({
            'units':
                self.units,
            'update_edge_features':
                self.update_edge_features,
            'batch_norm':
                self._batch_norm,
            'residual':
                self._residual,
            'dropout':
                self._dropout,
            'activation':
                activations.serialize(self._activation),
            'use_bias':
                self._use_bias,
            'kernel_initializer':
                initializers.serialize(self._kernel_initializer),
            'bias_initializer':
                initializers.serialize(self._bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self._kernel_regularizer),
            'bias_regularizer':
                regularizers.serialize(self._bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self._activity_regularizer),
            'kernel_constraint':
                constraints.serialize(self._kernel_constraint),
            'bias_constraint':
                constraints.serialize(self._bias_constraint),
            'node_feature_shape':
                self._node_feature_shape,
            'edge_feature_shape':
                self._edge_feature_shape,
        })
        return config
