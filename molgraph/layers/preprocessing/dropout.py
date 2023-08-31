import tensorflow as tf
from tensorflow import keras

from molgraph.internal import register_keras_serializable 

from molgraph.tensors.graph_tensor import GraphTensor


@register_keras_serializable(package='molgraph')
class NodeDropout(keras.layers.Layer):
    '''Randomly dropping nodes from the graph.

    Example usage:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     sizes=[2, 3],
    ...     node_feature=[[1., 0.], [1., 0.], [1., 0.], [1., 0.], [0., 1.]],
    ...     edge_feature=[[1., 0.], [0., 1.], [0., 1.], [0., 1.], 
    ...                   [1., 0.], [0., 1.], [1., 0.], [0., 1.]],
    ...     edge_src=[1, 0, 3, 4, 2, 4, 3, 2],
    ...     edge_dst=[0, 1, 2, 2, 3, 3, 4, 4],
    ... )
    >>> model = tf.keras.Sequential([
    ...     molgraph.layers.NodeDropout(rate=1.0),
    ...     molgraph.layers.EdgeDropout(rate=0.0)
    ... ])
    >>> model(graph_tensor).edge_feature.shape
    TensorShape([0, 2])

    Args:
        rate (float):
            The dropout rate. Default to 0.15.
    '''

    def __init__(self, rate: float = 0.15, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def call(self, tensor: GraphTensor) -> GraphTensor:
        '''Defines the computation from inputs to outputs.

        This method should not be called directly, but indirectly
        via ``__call__()``. Upon first call, the layer is automatically
        built via ``build()``.

        Args:
            tensor (GraphTensor):
                Input to the layer.

        Returns:
            GraphTensor: a instance of GraphTensor with some of its nodes dropped.
        '''
        if isinstance(tensor.node_feature, tf.RaggedTensor):
            shape = (tf.reduce_sum(tensor.node_feature.row_lengths()),)
        else:
            shape = tf.shape(tensor.node_feature)[:1]
        mask = tf.random.uniform(shape) > self.rate
        return tf.boolean_mask(tensor, mask, axis='node')

    def get_config(self):
        config = {'rate': self.rate,}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@register_keras_serializable(package='molgraph')
class EdgeDropout(NodeDropout):
    '''Randomly dropping edges from the graph.

    Example usage:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     sizes=[2, 3],
    ...     node_feature=[[1., 0.], [1., 0.], [1., 0.], [1., 0.], [0., 1.]],
    ...     edge_feature=[[1., 0.], [0., 1.], [0., 1.], [0., 1.], 
    ...                   [1., 0.], [0., 1.], [1., 0.], [0., 1.]],
    ...     edge_src=[1, 0, 3, 4, 2, 4, 3, 2],
    ...     edge_dst=[0, 1, 2, 2, 3, 3, 4, 4],
    ... )
    >>> model = tf.keras.Sequential([
    ...     molgraph.layers.NodeDropout(rate=0.0),
    ...     molgraph.layers.EdgeDropout(rate=0.15)
    ... ])
    >>> model(graph_tensor).node_feature.shape
    TensorShape([5, 2])

    Args:
        rate (float):
            The dropout rate. Default to 0.15.
    '''

    def call(self, tensor: GraphTensor) -> GraphTensor:
        '''Defines the computation from inputs to outputs.

        This method should not be called directly, but indirectly
        via ``__call__()``. Upon first call, the layer is automatically
        built via ``build()``.

        Args:
            tensor (GraphTensor):
                Input to the layer.

        Returns:
            GraphTensor: a instance of GraphTensor with some of its edges dropped.
        '''
        if isinstance(tensor.edge_src, tf.RaggedTensor):
            shape = (tf.reduce_sum(tensor.edge_src.row_lengths()),)
        else:
            shape = tf.shape(tensor.edge_src)[:1]
        mask = tf.random.uniform(shape) > self.rate
        return tf.boolean_mask(tensor, mask, axis='edge')

