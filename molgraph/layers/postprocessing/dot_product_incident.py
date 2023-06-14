import tensorflow as tf
from tensorflow import keras

from typing import Optional

from molgraph.tensors.graph_tensor import GraphTensor


@keras.utils.register_keras_serializable(package='molgraph')
class DotProductIncident(keras.layers.Layer):
    '''Performs dot product on the incident node features.

    Useful for e.g., edge and link classification.

    **Example:**

    >>> graph_tensor = molgraph.GraphTensor(
    ...     data={
    ...         'edge_src': [[1, 0], [1, 2, 0, 2, 1, 0]],
    ...         'edge_dst': [[0, 1], [0, 0, 1, 1, 2, 2]],
    ...         'node_feature': [
    ...             [[2.0, 0.0], [2.0, 0.0]],
    ...             [[3.0, 0.0], [3.0, 0.0], [0.0, 3.0]]
    ...         ],
    ...     }
    ... )
    >>> model = tf.keras.Sequential([
    ...     tf.keras.layers.Input(type_spec=graph_tensor.unspecific_spec),
    ...     molgraph.layers.DotProductIncident()
    ... ])
    >>> model(graph_tensor)
    GraphTensor(
      edge_src=<tf.RaggedTensor: shape=(2, None), dtype=int32>,
      edge_dst=<tf.RaggedTensor: shape=(2, None), dtype=int32>,
      node_feature=<tf.RaggedTensor: shape=(2, None, 2), dtype=float32>,
      edge_score=<tf.RaggedTensor: shape=(2, None, 1), dtype=float32>)

    Args:
        apply_sigmoid (bool):
            Whether to apply a sigmoid activaton on the edge scores. 
            Default to False.
        data_field (str, None):
            Name of the data added to the GraphTensor instance. If None,
            the output will be a ``tf.Tensor`` or ``tf.RaggedTensor`` 
            containing the dot product between incident node features.
            If str, a GraphTensor instance with a new data field "data_field"
            will be outputted. Default to "edge_score".
    '''
    def __init__(
        self, 
        apply_sigmoid: bool = False, 
        data_field: Optional[str] = 'edge_score',
        **kwargs
    ):
        super().__init__(**kwargs)
        self._apply_sigmoid = apply_sigmoid
        self._data_field = data_field

    def call(self, tensor: GraphTensor) -> GraphTensor:
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
        if isinstance(tensor.node_feature, tf.RaggedTensor):
            tensor = tensor.merge()
        
        adjacency = tf.stack([
            tensor.edge_src, tensor.edge_dst], axis=1)
        node_feature_incident = tf.gather(
            tensor.node_feature, adjacency)
        edge_score = tf.reduce_sum(
            tf.reduce_prod(node_feature_incident, axis=1), axis=1, keepdims=True)
        if self._apply_sigmoid:
            edge_score = tf.nn.sigmoid(edge_score)
        if self._data_field is None:
            return edge_score
        return tensor_orig.update({self._data_field: edge_score})

    def get_config(self):
        config = super().get_config()
        config.update({
            'apply_sigmoid': self._apply_sigmoid,
            'data_field': self._data_field,
        })
        return config
    