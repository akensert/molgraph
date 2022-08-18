import tensorflow as tf
from tensorflow import keras

from molgraph.tensors.graph_tensor import GraphTensor


@keras.utils.register_keras_serializable(package='molgraph')
class GatherIncident(keras.layers.Layer):
    '''Gathers incident node features.

    Useful for e.g., downstream edge and link classification.

    **Example:**

    >>> graph_tensor = molgraph.GraphTensor(
    ...     data={
    ...         'edge_dst': [[0, 1], [0, 0, 1, 1, 2, 2]],
    ...         'edge_src': [[1, 0], [1, 2, 0, 2, 1, 0]],
    ...         'node_feature': [
    ...             [[1.0, 0.0], [2.0, 0.0]],
    ...             [[3.0, 0.0], [4.0, 0.0], [0.0, 5.0]]
    ...         ],
    ...     }
    ... )
    ... graph_tensor = graph_tensor.merge()
    >>> model = tf.keras.Sequential([
    ...     tf.keras.layers.Input(type_spec=graph_tensor.unspecific_spec),
    ...     molgraph.layers.GatherIncident(concat=True)
    ... ])
    >>> model(graph_tensor)
    <tf.Tensor: shape=(8, 4), dtype=float32, numpy=
    array([[1., 0., 2., 0.],
           [2., 0., 1., 0.],
           [3., 0., 4., 0.],
           [3., 0., 0., 5.],
           [4., 0., 3., 0.],
           [4., 0., 0., 5.],
           [0., 5., 4., 0.],
           [0., 5., 3., 0.]], dtype=float32)>

    Args:
        concat (bool):
            Whether to concatenate incident node features or not. If True,
            resulting shape is (num_edges, num_features * 2), if False,
            resulting shape is (num_edges, 2, num_features).
    '''

    def __init__(self, concat: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.concat = concat

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
        tensor_orig = tensor
        if isinstance(tensor.node_feature, tf.RaggedTensor):
            tensor = tensor.merge()
        node_feature_src = tf.gather(tensor.node_feature, tensor.edge_src)
        node_feature_dst = tf.gather(tensor.node_feature, tensor.edge_dst)
        if self.concat:
            node_feature_incident = tf.concat([
                node_feature_dst, node_feature_src], axis=1)
        else:
            node_feature_incident = tf.stack([
                node_feature_dst, node_feature_src], axis=1)
        tensor_orig = tensor_orig.update({
            'node_feature_incident': node_feature_incident})
        return tensor_orig.node_feature_incident

    def get_config(self):
        base_config = super().get_config()
        base_config.update({'concat': self.concat})
        return base_config
