import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
from keras.layers.preprocessing import preprocessing_utils as utils

from typing import Tuple
from typing import List
from typing import Optional
from typing import Union
from typing import Tuple


from molgraph.tensors.graph_tensor import GraphTensor



@keras.utils.register_keras_serializable(package='molgraph')
class CenterScaling(layers.experimental.preprocessing.PreprocessingLayer):

    '''Centering.

    Specify, as keyword argument only,
    ``CenterScaling(feature='node_feature')`` to perform standard scaling
    on the ``node_feature`` component of the ``GraphTensor``, or,
    ``CenterScaling(feature='edge_feature')`` to perform standard scaling
    on the ``edge_feature`` component of the ``GraphTensor``. If not specified,
    the ``node_feature`` component will be considered.

    Args:
        mean (tf.Tensor, None):
            The mean of the features. Default to None.
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
            data = data.map(lambda x: getattr(x, self.feature))
        else:
            data = getattr(data, self.feature)
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
                ``node_features`` component or the ``edge_features``
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
                components of ``GraphTensor``.
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
            batch_count += tf.reduce_sum(feature.row_lengths())
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
            'mean': utils.listify_tensors(self.mean),
            'feature': self.feature,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@keras.utils.register_keras_serializable(package='molgraph')
class NodeCenterScaling(CenterScaling):
    feature = 'node_feature'


@keras.utils.register_keras_serializable(package='molgraph')
class EdgeCenterScaling(CenterScaling):
    feature = 'edge_feature'