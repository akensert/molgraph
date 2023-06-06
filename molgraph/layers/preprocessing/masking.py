import tensorflow as tf
from tensorflow import keras

from molgraph.tensors.graph_tensor import GraphTensor


@keras.utils.register_keras_serializable(package='molgraph')
class FeatureMasking(keras.layers.Layer):
    '''Feature masking.

    Specify, as keyword argument only,
    ``FeatureMasking(feature='node_feature')`` to perform masking
    on the ``node_feature`` field of the ``GraphTensor``, or,
    ``FeatureMasking(feature='edge_feature')`` to perform masking
    on the ``edge_feature`` field of the ``GraphTensor``. If not specified,
    the ``node_feature`` field will be considered.

    **Example:**

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
    ...         'edge_feature': [
    ...             [1.0, 0.0],
    ...             [0.0, 1.0],
    ...             [0.0, 1.0],
    ...             [0.0, 1.0],
    ...             [1.0, 0.0],
    ...             [0.0, 1.0],
    ...             [1.0, 0.0],
    ...             [0.0, 1.0]
    ...         ],
    ...         'graph_indicator': [0, 0, 1, 1, 1],
    ...     }
    ... )
    >>> model = tf.keras.Sequential([
    ...     tf.keras.layers.Input(type_spec=graph_tensor.unspecific_spec),
    ...     molgraph.layers.FeatureMasking(
    ...         feature='node_feature', mask_freq=0.15),
    ...     molgraph.layers.FeatureMasking(
    ...         feature='edge_feature', mask_freq=0.15)
    ... ])
    >>> model.output_shape
    (None, 2)

    Args:
        mask_freq (float):
            The frequency of the masking. Default to 0.15.
        **kwargs:
            Specify the relevant ``feature``. Default to ``node_feature``.
            The reminaing kwargs are passed to the parent class.
    '''

    def __init__(
        self,
        mask_freq: float = 0.15,
        **kwargs
    ):
        if 'feature' in kwargs:
            self.feature = kwargs.pop('feature')
        elif not hasattr(self, 'feature'):
            self.feature = 'node_feature'
        super().__init__(**kwargs)
        self.mask_freq = mask_freq

    def call(self, tensor: GraphTensor) -> GraphTensor:
        '''Defines the computation from inputs to outputs.

        This method should not be called directly, but indirectly
        via ``__call__()``. Upon first call, the layer is automatically
        built via ``build()``.

        Args:
            tensor (GraphTensor):
                Input to the layer.

        Returns:
            A ``tf.Tensor`` or `tf.RaggedTensor` based on the ``node_feature``
            field of the inputted ``GraphTensor``.
        '''
        feature = getattr(tensor, self.feature)
        if isinstance(feature, tf.RaggedTensor):
            shape = (tf.reduce_sum(feature.row_lengths()),)
        else:
            shape = tf.shape(feature)[:1]
        mask = tf.random.uniform(shape) > self.mask_freq
        return tf.boolean_mask(tensor, mask, axis=self.feature[:4])

    def get_config(self):
        config = {
            'feature': self.feature,
            'mask_freq': self.mask_freq,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@keras.utils.register_keras_serializable(package='molgraph')
class NodeFeatureMasking(FeatureMasking):
    feature = 'node_feature'


@keras.utils.register_keras_serializable(package='molgraph')
class EdgeFeatureMasking(FeatureMasking):
    feature = 'edge_feature'
