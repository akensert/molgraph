import tensorflow as tf
from tensorflow import keras

from molgraph.tensors.graph_tensor import GraphTensor


@keras.utils.register_keras_serializable(package='molgraph')
class ExtractField(keras.layers.Layer):
    '''Extract specific field of ``GraphTensor``.

    **Example:**

    >>> graph_tensor = molgraph.GraphTensor(
    ...     data={
    ...         'edge_dst': [0, 1, 2, 2, 3, 3, 4, 4],
    ...         'edge_src': [1, 0, 3, 4, 2, 4, 3, 2],
    ...         'node_feature': [
    ...             [1., 0.],
    ...             [2., 0.],
    ...             [3., 0.],
    ...             [4., 0.],
    ...             [0., 5.]
    ...         ],
    ...         'graph_indicator': [0, 0, 1, 1, 1],
    ...     }
    ... )
    >>> model = tf.keras.Sequential([
    ...     tf.keras.layers.Input(type_spec=graph_tensor.unspecific_spec),
    ...     molgraph.layers.ExtractField('node_feature')
    ... ])
    >>> model(graph_tensor)
    <tf.Tensor: shape=(5, 2), dtype=float32, numpy=
    array([[1., 0.],
           [2., 0.],
           [3., 0.],
           [4., 0.],
           [0., 5.]], dtype=float32)>

    Args:
        field (str):
            Field to extract from ``GraphTensor``.
    '''
    def __init__(self, field, **kwargs):
        super().__init__(**kwargs)
        self.field = field

    def call(self, tensor: GraphTensor) -> tf.Tensor:
        '''Defines the computation from inputs to outputs.

        This method should not be called directly, but indirectly
        via ``__call__()``. Upon first call, the layer is automatically
        built via ``build()``.

        Args:
            tensor (GraphTensor):
                Input to the layer.

        Returns:
            A ``tf.Tensor`` or `tf.RaggedTensor` based on the node_feature
            field of the inputted ``GraphTensor``.
        '''
        return tensor[self.field]

    def get_config(self):
        base_config = super().get_config()
        base_config.update({'field': field})
        return base_config
