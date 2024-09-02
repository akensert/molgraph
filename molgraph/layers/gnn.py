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
