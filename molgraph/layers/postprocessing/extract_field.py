import tensorflow as tf
from tensorflow import keras

from molgraph.internal import register_keras_serializable 

from molgraph.tensors.graph_tensor import GraphTensor


@register_keras_serializable(package='molgraph')
class ExtractField(keras.layers.Layer):
    '''Extract specific field of ``GraphTensor``.

    Example usage:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     sizes=[2, 3],
    ...     node_feature=[[1., 0.], [1., 0.], [1., 0.], [1., 0.], [0., 1.]],
    ...     edge_src=[1, 0, 3, 4, 2, 4, 3, 2],
    ...     edge_dst=[0, 1, 2, 2, 3, 3, 4, 4],
    ... )
    >>> model = tf.keras.Sequential([
    ...     molgraph.layers.ExtractField('node_feature')
    ... ])
    >>> model(graph_tensor)
    <tf.Tensor: shape=(5, 2), dtype=float32, numpy=
    array([[1., 0.],
           [1., 0.],
           [1., 0.],
           [1., 0.],
           [0., 1.]], dtype=float32)>

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
        return getattr(tensor, self.field)

    def get_config(self):
        base_config = super().get_config()
        base_config.update({'field': self.field})
        return base_config
