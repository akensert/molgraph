import tensorflow as tf
from tensorflow import keras

from molgraph.internal import register_keras_serializable 
from molgraph.tensors.graph_tensor import GraphTensor 


@register_keras_serializable(package='molgraph')
class UpdateField(keras.layers.Layer):

    def __init__(
        self, 
        field: str = 'node_feature', 
        func: keras.Model = None, 
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.field = field
        self._apply_func = func 
    
    def call(self, tensor: GraphTensor) -> GraphTensor:
        return tensor.update({
            self.field: self._apply_func(getattr(tensor, self.field))})
    
    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'field': self.field,
            'func': keras.saving.serialize_keras_object(self._apply_func)})
        return config

    @classmethod
    def from_config(cls, config: dict) -> 'UpdateField':
        config['func'] = keras.saving.deserialize_keras_object(config['func'])
        return cls(**config)