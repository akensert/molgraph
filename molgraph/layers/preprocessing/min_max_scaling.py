import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers

from keras import backend
from keras.layers.preprocessing import preprocessing_utils as utils

from typing import Tuple
from typing import List
from typing import Optional
from typing import Union
from typing import Tuple


from molgraph.tensors.graph_tensor import GraphTensor


@keras.utils.register_keras_serializable(package='molgraph')
class MinMaxScaling(layers.experimental.preprocessing.PreprocessingLayer):

    '''Min-max scaling between a specified range.

    Specify, as keyword argument only,
    ``MinMaxScaling(feature='node_feature')`` to perform standard scaling
    on the ``node_feature`` field of the ``GraphTensor``, or,
    ``MinMaxScaling(feature='edge_feature')`` to perform standard scaling
    on the ``edge_feature`` field of the ``GraphTensor``. If not specified,
    the ``node_feature`` field will be considered.

    **Examples:**

    Adapt layer on ``GraphTensor`` directly:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     data={
    ...         'edge_dst': [[0, 1], [0, 0, 1, 1, 2, 2]],
    ...         'edge_src': [[1, 0], [1, 2, 0, 2, 1, 0]],
    ...         'node_feature': [
    ...             [[2.0, 0.5], [2.0, 0.0]],
    ...             [[2.0, 0.0], [2.0, 0.5], [0.0, 2.0]]
    ...         ],
    ...     }
    ... )
    >>> # Initialize layer
    >>> preprocessing = molgraph.layers.MinMaxScaling(
    ...    feature='node_feature')
    >>> # Adapt layer to graph_tensor
    >>> preprocessing.adapt(graph_tensor)
    >>> model = tf.keras.Sequential([
    ...     tf.keras.layers.Input(type_spec=graph_tensor.unspecific_spec),
    ...     preprocessing,
    ... ])
    >>> graph_tensor = model(graph_tensor)
    >>> graph_tensor.merge().node_feature
    <tf.Tensor: shape=(5, 2), dtype=float32, numpy=
    array([[1.  , 0.25],
           [1.  , 0.  ],
           [1.  , 0.  ],
           [1.  , 0.25],
           [0.  , 1.  ]], dtype=float32)>

    Adapt layer on ``tf.data.Dataset`` constructed from ``GraphTensor``:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     data={
    ...         'edge_dst': [[0, 1], [0, 0, 1, 1, 2, 2]],
    ...         'edge_src': [[1, 0], [1, 2, 0, 2, 1, 0]],
    ...         'node_feature': [
    ...             [[2.0, 0.5], [2.0, 0.0]],
    ...             [[2.0, 0.0], [2.0, 0.5], [0.0, 2.0]]
    ...         ],
    ...     }
    ... )
    >>> # Obtain dataset
    >>> ds = tf.data.Dataset.from_tensor_slices(graph_tensor).batch(2)
    >>> # Initialize layer
    >>> preprocessing = molgraph.layers.MinMaxScaling(
    ...    feature='node_feature')
    >>> # Adapt layer to graph_tensor
    >>> preprocessing.adapt(ds)
    >>> model = tf.keras.Sequential([
    ...     tf.keras.layers.Input(type_spec=graph_tensor.unspecific_spec),
    ...     preprocessing,
    ... ])
    >>> output = model.predict(ds)
    >>> output.merge().node_feature
    <tf.Tensor: shape=(5, 2), dtype=float32, numpy=
    array([[1.  , 0.25],
           [1.  , 0.  ],
           [1.  , 0.  ],
           [1.  , 0.25],
           [0.  , 1.  ]], dtype=float32)>

    Adapt layer on merged ``GraphTensor``:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     data={
    ...         'edge_dst': [0, 1, 2, 2, 3, 3, 4, 4],
    ...         'edge_src': [1, 0, 3, 4, 2, 4, 3, 2],
    ...         'node_feature': [
    ...             [2.0, 0.5],
    ...             [2.0, 0.0],
    ...             [2.0, 0.0],
    ...             [2.0, 0.5],
    ...             [0.0, 2.0]
    ...         ],
    ...         'graph_indicator': [0, 0, 1, 1, 1],
    ...     }
    ... )
    >>> # Initialize layer
    >>> preprocessing = molgraph.layers.MinMaxScaling(
    ...    feature='node_feature')
    >>> # Adapt layer to graph_tensor
    >>> preprocessing.adapt(graph_tensor)
    >>> model = tf.keras.Sequential([
    ...     tf.keras.layers.Input(type_spec=graph_tensor.unspecific_spec),
    ...     preprocessing,
    ... ])
    >>> graph_tensor = model(graph_tensor)
    >>> graph_tensor.node_feature
    <tf.Tensor: shape=(5, 2), dtype=float32, numpy=
    array([[1.  , 0.25],
           [1.  , 0.  ],
           [1.  , 0.  ],
           [1.  , 0.25],
           [0.  , 1.  ]], dtype=float32)>

    Args:
        feature_range (tuple):
            The range of values of transformed data.
        minimum (tf.Tensor, None):
            The minimum values of the original data; used to transform the data
            into the given feature_range. If None, the layer has to be adapted
            via e.g. `adapt()`. Default to None.
        maximum (tf.Tensor, None):
            The maximum values of the original data; used to transform the data
            into the given feature_range. If None, the layer has to be adapted
            via e.g. `adapt()`. Default to None.
        **kwargs:
            Specify the relevant ``feature``. Default to ``node_feature``.
            The reminaing kwargs are passed to the parent class.
    '''

    def __init__(
        self,
        feature_range: Tuple[float, float] = (0, 1),
        minimum: Optional[tf.Tensor] = None,
        maximum: Optional[tf.Tensor] = None,
        threshold: bool = True,
        **kwargs
    ):
        if 'feature' in kwargs:
            self.feature = kwargs.pop('feature')
        elif not hasattr(self, 'feature'):
            self.feature = 'node_feature'

        super().__init__(**kwargs)
        self.input_minimum = minimum
        self.input_maximum = maximum
        self.threshold = threshold
        self.feature_range = feature_range

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
            data = data = data.map(
                lambda x: getattr(x, self.feature))
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
                ``node_features`` field or the ``edge_features``
                field (of the ``GraphTensor``) are updated.
        '''
        feature = getattr(data, self.feature)

        if isinstance(feature, tf.Tensor):
            gather_axis = 1
            broadcast_shape = (1,) + self.minimum.shape
        else:
            gather_axis = 2
            broadcast_shape = (1, 1) + self.minimum.shape

        minimum = tf.reshape(self.minimum, broadcast_shape)
        maximum = tf.reshape(self.maximum, broadcast_shape)
        feature = (
            tf.math.divide_no_nan((feature - minimum), (maximum - minimum)) *
            (self.feature_range[1] - self.feature_range[0]) +
            self.feature_range[0])

        if self.threshold:
            feature = tf.gather(feature, self.keep, axis=gather_axis)

        return data.update({self.feature: feature})

    def build(self, input_shape):

        super().build(input_shape)

        self.adapt_maximum = self.add_weight(
            name='maximum',
            shape=input_shape[-1:],
            dtype=tf.float32,
            initializer=initializers.Constant(-float('inf')),
            trainable=False)

        self.adapt_minimum = self.add_weight(
            name='minimum',
            shape=input_shape[-1:],
            dtype=tf.float32,
            initializer=initializers.Constant(float('inf')),
            trainable=False)

        if self.threshold:

            self.adapt_keep = self.add_weight(
                name='keep',
                shape=input_shape[-1:],
                dtype=tf.bool,
                initializer=initializers.Constant(True),
                trainable=False)

        if self.input_minimum is None:
            self.finalize_state()
        else:
            self.adapt_maximum.assign(self.input_maximum)
            self.adapt_minimum.assign(self.input_minimum)

            self.maximum = self.adapt_maximum
            self.minimum = self.adapt_minimum

            if self.threshold:
                self.adapt_keep.assign(
                    tf.where((self.maximum - self.minimum) > 0.0, True, False))

                self.keep = tf.where(self.adapt_keep == True)[:, 0]

    def update_state(self, feature):

        if self.input_minimum is not None:
            raise ValueError("Cannot adapt")

        if isinstance(feature, tf.Tensor):
            axis = [0]
        else:
            axis = [0, 1]

        minimum = tf.math.reduce_min(feature, axis=axis)
        maximum = tf.math.reduce_max(feature, axis=axis)

        minimum = tf.minimum(minimum, self.adapt_minimum)
        maximum = tf.maximum(maximum, self.adapt_maximum)

        self.adapt_minimum.assign(minimum)
        self.adapt_maximum.assign(maximum)
        if self.threshold:
            self.adapt_keep.assign(tf.where(self.adapt_maximum - self.adapt_minimum > 0, True, False))

    def reset_state(self):
        if self.input_minimum is not None or not self.built:
            return
        self.adapt_minimum.assign(tf.zeros_like(self.adapt_minimum) + float('inf'))
        self.adapt_maximum.assign(tf.zeros_like(self.adapt_maximum) - float('inf'))
        if self.threshold:
            self.adapt_keep.assign(tf.ones_like(self.adapt_keep))

    def finalize_state(self):
        if self.input_minimum is not None or not self.built:
            return
        self.minimum = self.adapt_minimum
        self.maximum = self.adapt_maximum
        if self.threshold:
            self.keep = tf.where(self.adapt_keep == True)[:, 0]

    def get_config(self):
        config = {
            'feature': self.feature,
            'feature_range': self.feature_range,
            'minimum': utils.listify_tensors(self.minimum),
            'maximum': utils.listify_tensors(self.maximum),
            'threshold': self.threshold
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if self.variance_threshold is not None:
            return input_shape[:-1] + (len(self.keep),)
        return input_shape

    def compute_output_signature(self, input_spec):
        if self.variance_threshold is not None:
            shape = input_spec.shape[:-1]
            shape += (len(self.keep),)
            return tf.TensorSpec(shape, dtype=tf.float32)
        return input_spec


@keras.utils.register_keras_serializable(package='molgraph')
class NodeMinMaxScaling(MinMaxScaling):
    feature = 'node_feature'


@keras.utils.register_keras_serializable(package='molgraph')
class EdgeMinMaxScaling(MinMaxScaling):
    feature = 'edge_feature'
