import tensorflow as tf
from tensorflow import keras

from typing import Optional
from typing import Union
from typing import Tuple

from molgraph.internal import register_keras_serializable 

from molgraph.tensors.graph_tensor import GraphTensor


@register_keras_serializable(package='molgraph')
class DotProductIncident(keras.layers.Layer):
    '''Performs dot product on the incident node features.

    Useful for e.g., edge and link classification.

    Example usage:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     sizes=[2, 3],
    ...     node_feature=[[1., 0.], [1., 0.], [1., 0.], [1., 0.], [0., 1.]],
    ...     edge_src=[1, 0, 3, 4, 2, 4, 3, 2],
    ...     edge_dst=[0, 1, 2, 2, 3, 3, 4, 4],
    ... )
    >>> model = tf.keras.Sequential([
    ...     molgraph.layers.DotProductIncident()
    ... ])
    >>> model(graph_tensor)
    GraphTensor(
      sizes=<tf.Tensor: shape=(2,), dtype=int32>,
      node_feature=<tf.Tensor: shape=(5, 2), dtype=float32>,
      edge_src=<tf.Tensor: shape=(8,), dtype=int32>,
      edge_dst=<tf.Tensor: shape=(8,), dtype=int32>,
      edge_score=<tf.Tensor: shape=(8, 1), dtype=float32>)

    Args:
        normalize (bool):
            Whether to apply normalization on the edge scores. Produces cosine
            similarity values in the range -1 to 1. Default to False.
        axes (int, tuple):
            The axes (or axis) to perform the dot product. Default to 1.
        data_field (str, None):
            Name of the data added to the GraphTensor instance. If None,
            the output will be a ``tf.Tensor`` or ``tf.RaggedTensor`` 
            containing the dot product between incident node features.
            If str, a GraphTensor instance with a new data field "data_field"
            will be outputted. Default to "edge_score".
    '''
    def __init__(
        self, 
        normalize: bool = False, 
        axes: Union[int, Tuple[int, ...]] = 1,
        data_field: Optional[str] = 'edge_score',
        **kwargs
    ):
        super().__init__(**kwargs)
        self._dot_normalize = normalize
        self._dot_axes = axes
        self._data_field = data_field
        self._dot_layer = keras.layers.Dot(axes=axes, normalize=normalize)

    def call(
        self, 
        tensor: GraphTensor
    ) -> Union[GraphTensor, tf.Tensor, tf.RaggedTensor]:
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
        tensor_orig = tensor

        if tensor.is_ragged():
            tensor = tensor.merge()
        
        node_feature_src = tf.gather(
            tensor.node_feature, tensor.edge_src)

        node_feature_dst = tf.gather(
            tensor.node_feature, tensor.edge_dst)

        edge_score = self._dot_layer([node_feature_src, node_feature_dst])

        if self._data_field is not None:
            return tensor_orig.update({self._data_field: edge_score})
        
        tensor_orig = tensor_orig.update({'edge_score': edge_score})
        return tensor_orig.edge_score

    def get_config(self):
        config = super().get_config()
        config.update({
            'normalize': self._dot_normalize,
            'axes': self._dot_axes,
            'data_field': self._data_field,
        })
        return config
    