import tensorflow as tf
from tensorflow import keras

from molgraph.tensors.graph_tensor import GraphTensor


@keras.utils.register_keras_serializable(package='molgraph')
class DotProductIncident(keras.layers.Layer):
    """Useful for edge or link classification"""

    def __init__(self, partition: bool = True) -> None:
        super().__init__()
        self.partition = partition

    def call(self, tensor: GraphTensor) -> GraphTensor:
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
