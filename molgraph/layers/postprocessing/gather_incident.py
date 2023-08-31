import tensorflow as tf
from tensorflow import keras

from molgraph.internal import register_keras_serializable 

from molgraph.tensors.graph_tensor import GraphTensor


@register_keras_serializable(package='molgraph')
class GatherIncident(keras.layers.Layer):
    '''Gathers incident node features.

    Useful for e.g., downstream edge and link classification.

    Example usage:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     sizes=[2, 3],
    ...     node_feature=[[1., 0.], [1., 0.], [1., 0.], [1., 0.], [0., 1.]],
    ...     edge_src=[1, 0, 3, 4, 2, 4, 3, 2],
    ...     edge_dst=[0, 1, 2, 2, 3, 3, 4, 4],
    ... )
    >>> model = tf.keras.Sequential([
    ...     molgraph.layers.GatherIncident(concat=True)
    ... ])
    >>> model(graph_tensor)
    <tf.Tensor: shape=(8, 4), dtype=float32, numpy=
    array([[1., 0., 1., 0.],
           [1., 0., 1., 0.],
           [1., 0., 1., 0.],
           [0., 1., 1., 0.],
           [1., 0., 1., 0.],
           [0., 1., 1., 0.],
           [1., 0., 0., 1.],
           [1., 0., 0., 1.]], dtype=float32)>

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
                node_feature_src, node_feature_dst], axis=1)
        else:
            node_feature_incident = tf.stack([
                node_feature_src, node_feature_dst], axis=1)
        tensor_orig = tensor_orig.update({
            'edge_feature_incident': node_feature_incident})
        return tensor_orig.edge_feature_incident

    def get_config(self):
        base_config = super().get_config()
        base_config.update({'concat': self.concat})
        return base_config
