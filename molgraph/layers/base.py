import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.keras import activations

from keras.utils import tf_utils

from abc import ABC, abstractmethod

from typing import Optional
from typing import Callable
from typing import Union
from typing import Tuple

from molgraph.tensors.graph_tensor import GraphTensor


@keras.utils.register_keras_serializable(package='molgraph')
class _BaseLayer(layers.Layer, ABC):

    """Base layer for all (existing) graph convolutional layers.
    """

    def __init__(
        self,
        units: Optional[int] = None,
        batch_norm: bool = False,
        residual = False,
        dropout: Optional[float] = None,
        activation: Union[None, str, Callable[[tf.Tensor], tf.Tensor]] = None,
        use_bias: bool = False,
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
    ) -> None:

        self.update_edge_features = kwargs.pop('update_edge_features', False)
        kwargs.pop('use_edge_features', False)

        layers.Layer.__init__(self, **kwargs)

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
        self._built_from_signature = False
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

        tensor_orig = tensor
        if isinstance(tensor.node_feature, tf.RaggedTensor):
            tensor = tensor.merge()

        if not self._built_from_signature:
            self._build_from_signature(
                getattr(tensor, 'node_feature', None),
                getattr(tensor, 'edge_feature', None)
            )

        tensor_update = self.subclass_call(tensor)
        tensor_update = self._process_output(tensor_update, tensor)

        return tensor_orig.update({
            k: v for (k, v) in tensor_update._data.items() if
            k not in ['edge_dst', 'edge_src', 'graph_indicator']
        })

    def _build_from_signature(
        self,
        node_feature: Union[tf.Tensor, tf.TensorShape],
        edge_feature: Optional[Union[tf.Tensor, tf.TensorShape]] = None
    ) -> None:

        self._built_from_signature = True

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

    def get_kernel(self, shape, dtype=tf.float32, name='kernel'):
        return self.add_weight(
            name=name,
            shape=shape,
            dtype=dtype,
            initializer=self._kernel_initializer,
            regularizer=self._kernel_regularizer,
            constraint=self._kernel_constraint,
            trainable=True
        )

    def get_bias(self, shape, dtype=tf.float32, name='bias'):
        return self.add_weight(
            name=name,
            shape=shape,
            dtype=dtype,
            initializer=self._bias_initializer,
            regularizer=self._bias_regularizer,
            constraint=self._bias_constraint,
            trainable=True
        )

    def get_dense(
        self, units: int,
        activation: Union[None, str, Callable[[tf.Tensor], tf.Tensor]] = None,
    ) -> Callable[[tf.Tensor], tf.Tensor]:
        return layers.Dense(
            units,
            activation=activation,
            use_bias=self._use_bias,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint
        )

    def get_einsum_dense(
        self,
        equation: str,
        output_shape: Tuple[int, ...],
        activation: Union[None, str, Callable[[tf.Tensor], tf.Tensor]] = None,
    ) -> Callable[[tf.Tensor], tf.Tensor]:

        if self._use_bias:
            bias_axes = equation.split('->')[-1][1:]
            if len(bias_axes) == 0:
                bias_axes = None
        else:
            bias_axes = None

        return layers.experimental.EinsumDense(
            equation,
            output_shape,
            activation=activation,
            bias_axes=bias_axes,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint
        )

    def compute_output_shape(self, input_shape):
        inner_dim = self.units
        if getattr(self, 'merge_mode', None) == 'concat':
            if hasattr(self, 'num_heads'):
                inner_dim *= self.num_heads
            elif hasattr(self, 'num_kernels'):
                inner_dim *= getattr(self, 'num_kernels', 1)
        return tf.TensorShape(input_shape[:-1]).concatenate([inner_dim])

    @classmethod
    def from_config(cls, config):
        node_feature_shape = config.pop('node_feature_shape')
        edge_feature_shape = config.pop('edge_feature_shape')
        layer = cls(**config)
        if None in [node_feature_shape, edge_feature_shape]:
            pass # TODO(akensert): add warning message about not restoring weights
        else:
            layer._build_from_signature(node_feature_shape, edge_feature_shape)
        return layer

    def get_config(self):
        """Returns the config of the layer.

        Returns
        -------
        Python dictionary.
        """
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
