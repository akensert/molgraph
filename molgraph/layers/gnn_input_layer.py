import keras

from molgraph.internal import register_keras_serializable 
from molgraph.tensors.graph_tensor import GraphTensor 


@register_keras_serializable(package='molgraph')
class GNNInputLayer(keras.layers.InputLayer):
    
    def get_config(self):
        config = super().get_config()
        type_spec = config.pop('type_spec').__dict__
        type_spec['auxiliary'] = dict(type_spec['auxiliary'])
        config['type_spec'] = type_spec
        return config
    
    @classmethod
    def from_config(cls, config):
        type_spec = config.pop('type_spec')
        type_spec = keras.saving.deserialize_keras_object(type_spec)
        auxiliary_spec = type_spec.pop('auxiliary')
        type_spec = {**type_spec, **auxiliary_spec}
        config['type_spec'] = GraphTensor.Spec(**type_spec)
        return cls(**config)