import tensorflow as tf
from tensorflow import keras
from keras import layers

import numpy as np

from typing import Optional

from molgraph.internal import register_keras_serializable 
from molgraph.internal import PreprocessingLayer

from molgraph.tensors.graph_tensor import GraphTensor


@register_keras_serializable(package='molgraph')
class CenterScaling(PreprocessingLayer):

    '''Centering.

    Specify, as keyword argument only,
    ``CenterScaling(feature='node_feature')`` to perform center scaling
    on the ``node_feature`` component of the ``GraphTensor``, or,
    ``CenterScaling(feature='edge_feature')`` to perform center scaling
    on the ``edge_feature`` component of the ``GraphTensor``. If not specified,
    the ``node_feature`` component will be considered.

    Example usage:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     sizes=[2, 3],
    ...     node_feature=[[1., 0.], [1., 0.], [1., 0.], [1., 0.], [0., 1.]],
    ...     edge_src=[1, 0, 3, 4, 2, 4, 3, 2],
    ...     edge_dst=[0, 1, 2, 2, 3, 3, 4, 4],
    ... )
    >>> preprocessing = molgraph.layers.CenterScaling(
    ...    feature='node_feature')
    >>> preprocessing.adapt(graph_tensor)
    >>> model = tf.keras.Sequential([preprocessing])
    >>> graph_tensor = model(graph_tensor)
    >>> graph_tensor.node_feature
    <tf.Tensor: shape=(5, 2), dtype=float32, numpy=
    array([[ 0.19999999, -0.2       ],
           [ 0.19999999, -0.2       ],
           [ 0.19999999, -0.2       ],
           [ 0.19999999, -0.2       ],
           [-0.8       ,  0.8       ]], dtype=float32)>

    Adapt layer on a tf.data.Dataset:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     sizes=[2, 3],
    ...     node_feature=[[1., 0.], [1., 0.], [1., 0.], [1., 0.], [0., 1.]],
    ...     edge_src=[1, 0, 3, 4, 2, 4, 3, 2],
    ...     edge_dst=[0, 1, 2, 2, 3, 3, 4, 4],
    ... )
    >>> ds = tf.data.Dataset.from_tensor_slices(graph_tensor).batch(2)
    >>> preprocessing = molgraph.layers.CenterScaling(
    ...    feature='node_feature')
    >>> preprocessing.adapt(ds)
    >>> model = tf.keras.Sequential([preprocessing])
    >>> output = model.predict(ds, verbose=0)
    >>> output.node_feature
    <tf.Tensor: shape=(5, 2), dtype=float32, numpy=
    array([[ 0.19999999, -0.2       ],
           [ 0.19999999, -0.2       ],
           [ 0.19999999, -0.2       ],
           [ 0.19999999, -0.2       ],
           [-0.8       ,  0.8       ]], dtype=float32)>

    Args:
        mean (tf.Tensor, None):
            The means of the original data; used to transform the data.
            If None, the layer has to be adapted via e.g. `adapt()`.
            Default to None.
        **kwargs:
            Specify the relevant ``feature``. Default to ``node_feature``.
            The reminaing kwargs are passed to the parent class.
    '''

    def __init__(
        self,
        mean: Optional[tf.Tensor] = None,
        **kwargs
    ):
        if 'feature' in kwargs:
            self.feature = kwargs.pop('feature')
        elif not hasattr(self, 'feature'):
            self.feature = 'node_feature'

        super().__init__(**kwargs)
        self.input_mean = mean

    def adapt(self, data, batch_size=None, steps=None):
        '''Adapts the layer to data.

        When adapting the layer to the data, ``build()`` will be called
        automatically (to initialize the relevant attributes). After adaption,
        the layer is finalized and ready to be used.

        Args:
            data (GraphTensor, tf.data.Dataset):
                Data to be used to adapt the layer. Can be either a
                ``GraphTensor`` directly or a ``tf.data.Dataset`` constructed
                from a ``GraphTensor``.
            batch_size (int, None):
                The batch size to be used during adaption. Default to None.
            steps (int, None):
                The number of steps of adaption. If None, the number of
                samples divided by the batch_size is used. Default to None.
        '''

        if not isinstance(data,  GraphTensor):
            data = data.map(
                lambda x: getattr(x, self.feature))
            for x in data.take(1):
                self.build(x.shape)
        else:
            data = getattr(data, self.feature)
            self.build(data.shape)
            
        super().adapt(data, batch_size=batch_size, steps=steps)

    def call(self, data):
        '''Defines the computation from inputs to outputs.

        This method should not be called directly, but indirectly
        via ``__call__()``. Upon first call, the layer is automatically
        built via ``build()``.

        Args:
            data (GraphTensor):
                Input to the layer.

        Returns:
            GraphTensor:
                A ``GraphTensor`` with updated features. Either the
                ``node_feature`` component or the ``edge_feature``
                component (of the ``GraphTensor``) are updated.
        '''
        feature = getattr(data, self.feature)

        if isinstance(feature, tf.RaggedTensor):
            gather_axis = 2
            broadcast_shape = (1, 1) + self.mean.shape # unnecessary?
        else:
            gather_axis = 1
            broadcast_shape = (1,) + self.mean.shape # unnecessary?

        mean = tf.reshape(self.mean, broadcast_shape)
        feature -= mean
        return data.update({self.feature: feature})

    def build(self, input_shape):
        '''Builds the layer.

        Specifically, it initializes the ``mean`` to be adapted via ``adapt()``.
        Or if ``mean`` was supplied directly to the layer, ``adapt()`` can be
        ignored.

        Args:
            input_shape (list, tuple, tf.TensorShape):
                The shape of the input to the layer. Corresponds to either
                the ``node_feature`` component or the ``edge_feature``
                component of ``GraphTensor``.
        '''
        super().build(input_shape)

        self.adapt_mean = self.add_weight(
            name='mean',
            shape=input_shape[-1:],
            dtype=tf.float32,
            initializer='zeros',
            trainable=False)

        if self.input_mean is None:

            self.count = self.add_weight(
                name='count',
                shape=(),
                dtype=tf.int64,
                initializer='zeros',
                trainable=False)

            self.finalize_state()

        else:
            self.adapt_mean.assign(self.input_mean)
            self.mean = self.adapt_mean

    def update_state(self, feature):
        '''Accumulates statistics for the preprocessing layer.

        Args:
            feature (tf.Tensor, tf.RaggedTensor):
                A mini-batch of inputs to the layer. Corresponds to either
                the ``node_feature`` or ``edge_feature`` component of
                ``GraphTensor``.
        '''

        if self.input_mean is not None:
            raise ValueError("Cannot adapt")

        if isinstance(feature, tf.Tensor):
            batch_count = tf.shape(feature)[0]
            axis = [0]
        else:
            batch_count = tf.shape(feature.to_tensor())[0]
            batch_count += tf.cast(
                tf.reduce_sum(feature.row_lengths()), dtype=batch_count.dtype)
            axis = [0, 1]

        batch_count = tf.cast(batch_count, tf.int64)

        total_count = batch_count + self.count

        batch_weight = tf.cast(
            batch_count / total_count, dtype=self._dtype)

        existing_weight = 1. - batch_weight

        # mean
        batch_mean = tf.math.reduce_mean(feature, axis)

        total_mean = (
            self.adapt_mean * existing_weight + batch_mean * batch_weight)

        self.adapt_mean.assign(total_mean)

        self.count.assign(total_count)


    def reset_state(self):
        '''Resets the statistics of the preprocessing layer.
        '''
        if self.input_mean is not None or not self.built:
            return
        self.adapt_mean.assign(tf.zeros_like(self.adapt_mean))
        self.count.assign(tf.zeros_like(self.count))

    def finalize_state(self):
        '''Finalize the statistics for the preprocessing layer.

        This method is called at the end of adapt or after restoring a
        serialized preprocessing layer’s state.
        '''
        if self.input_mean is not None or not self.built:
            return
        self.mean = self.adapt_mean

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_output_signature(self, input_spec):
        return input_spec

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        config = {
            'mean': _listify_tensors(self.mean),
            'feature': self.feature,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@register_keras_serializable(package='molgraph')
class NodeCenterScaling(CenterScaling):
    feature = 'node_feature'


@register_keras_serializable(package='molgraph')
class EdgeCenterScaling(CenterScaling):
    feature = 'edge_feature'


def _listify_tensors(x):
    if tf.is_tensor(x):
        x = x.numpy()
    if isinstance(x, np.ndarray):
        x = x.tolist()
    return x