import tensorflow as tf
from tensorflow import keras

from molgraph.tensors.graph_tensor import GraphTensor


@keras.utils.register_keras_serializable(package='molgraph')
class Gather(keras.layers.Layer):

    def call(self, tensor: GraphTensor) -> tf.Tensor:
        return tensor.node_feature
