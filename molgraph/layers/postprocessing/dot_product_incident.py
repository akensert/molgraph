import tensorflow as tf
from tensorflow import keras

from molgraph.tensors.graph_tensor import GraphTensor


@keras.utils.register_keras_serializable(package='molgraph')
class DotProductIncident(keras.layers.Layer):
    '''Performs dot product on the incident node features.

    Useful for e.g., edge and link classification.

    **Example:**

    >>> graph_tensor = molgraph.GraphTensor(
    ...     data={
    ...         'edge_dst': [[0, 1], [0, 0, 1, 1, 2, 2]],
    ...         'edge_src': [[1, 0], [1, 2, 0, 2, 1, 0]],
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
    <tf.RaggedTensor [[4.0, 4.0], [9.0, 0.0, 9.0, 0.0, 0.0, 0.0]]>
    '''

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
            tensor.edge_dst, tensor.edge_src], axis=1)
        node_feature_incident = tf.gather(
            tensor.node_feature, adjacency)
        tensor_orig = tensor_orig.update({'edge_score': tf.reduce_sum(
            tf.reduce_prod(node_feature_incident, axis=1), axis=1)})
        return tensor_orig.edge_score
