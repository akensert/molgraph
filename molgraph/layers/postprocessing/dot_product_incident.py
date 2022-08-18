import tensorflow as tf
from tensorflow import keras

from molgraph.tensors.graph_tensor import GraphTensor


@keras.utils.register_keras_serializable(package='molgraph')
class DotProductIncident(keras.layers.Layer):
    '''Performs dot product on the incident node features.

    Useful for e.g., edge and link classification.

    Args:
        partition (bool):
            Whether to partition the output. I.e., whether the output should
            correspond to a single (disjoint) graph (nested tensors) or
            subgraphs (nested ragged tensors).
    '''

    def __init__(self, partition: bool = True) -> None:
        super().__init__()
        self.partition = partition

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
            component of the inputted ``GraphTensor``.
        '''
        adjacency = tf.stack([
            tensor.edge_dst, tensor.edge_src], axis=1)
        node_feature_incident = tf.gather(
            tensor.node_feature, adjacency)
        tensor = tensor.update({'edge_score': tf.reduce_sum(
            tf.reduce_prod(node_feature_incident, axis=1), axis=1)})
        if self.partition:
            return partition_edges(tensor)
        return tensor

    def get_config(self):
        base_config = super().get_config()
        config = {
            'partition': self.partition,
        }
        base_config.update(config)
        return base_config

def partition_edges(tensor, output_type='tensor'):
    graph_indicator_edges = tf.gather(
        tensor.graph_indicator, tensor.edge_dst)
    return tf.RaggedTensor.from_value_rowids(
        tensor.edge_score,
        graph_indicator_edges,
        nrows=tf.reduce_max(tensor.graph_indicator) + 1)
