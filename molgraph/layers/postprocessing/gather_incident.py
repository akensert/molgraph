import tensorflow as tf
from tensorflow import keras

from molgraph.tensors.graph_tensor import GraphTensor


@keras.utils.register_keras_serializable(package='molgraph')
class GatherIncident(keras.layers.Layer):
    """Useful for edge or link classification"""

    def __init__(self, merge_mode: str = 'concat', **kwargs) -> None:
        super().__init__(**kwargs)
        self.merge_mode = merge_mode

    def call(self, tensor: GraphTensor) -> tf.Tensor:
        node_feature_src = tf.gather(tensor.node_feature, tensor.edge_src)
        node_feature_dst = tf.gather(tensor.node_feature, tensor.edge_dst)
        # if concat: shape=(n_edges, n_features * 2)
        # if stack:  shape=(n_edges, 2, n_features)
        node_feature_incident = getattr(tf, self.merge_mode)([
            node_feature_dst, node_feature_src], axis=1)
        return node_feature_incident

    def get_config(self):
        base_config = super().get_config()
        config = {
            'merge_mode': self.merge_mode,
        }
        base_config.update(config)
        return base_config
