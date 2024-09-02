import tensorflow as tf 
from tensorflow import keras

from molgraph.tensors.graph_tensor import GraphTensor
from molgraph.internal import register_keras_serializable 


@register_keras_serializable(package='molgraph')
class GNN(tf.keras.layers.Layer):
    
    def __init__(self, layers: list[keras.layers.Layer], **kwargs) -> None:
        super().__init__(**kwargs)
        self._graph_layers = layers

    def call(self, tensor: GraphTensor) -> GraphTensor:
        x = tensor 
        node_feature_list = []
        for layer in self._graph_layers:
            x = layer(x)
            node_feature_list.append(x.node_feature)
        return tensor.update({
            'node_feature': tf.concat(node_feature_list, axis=-1)})
    
    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'layers': [
                keras.layers.serialize(layer) for layer in self._graph_layers]})
        return config

    @classmethod
    def from_config(cls, config: dict) -> 'GNN':
        config['layers'] = [
            keras.layers.deserialize(layer) for layer in config['layers']]
        return super().from_config(config)

    def _taped_call(
        self,
        tensor: GraphTensor,
        tape: tf.GradientTape,
    ) -> tuple[GraphTensor, list[tf.Tensor]]:
        x = tensor
        node_feature_taped = []
        node_feature_list = []
        for layer in self._graph_layers:
            tape.watch(x.node_feature)
            node_feature_taped.append(_get_flat_node_feature(x))
            x = layer(x)
            node_feature_list.append(x.node_feature)
        tensor = tensor.update({
            'node_feature': tf.concat(node_feature_list, axis=-1)})
        return tensor, node_feature_taped


def _get_flat_node_feature(x: GraphTensor):
    node_feature = x.node_feature 
    if isinstance(node_feature, tf.RaggedTensor):
        node_feature = node_feature.flat_values
    return node_feature
