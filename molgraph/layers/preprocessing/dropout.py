import tensorflow as tf
from tensorflow import keras

from molgraph.tensors.graph_tensor import GraphTensor


@keras.saving.register_keras_serializable(package='molgraph')
class NodeDropout(keras.layers.Layer):
    '''Randomly dropping nodes from the graph.

    **Example:**

    >>> graph_tensor = molgraph.GraphTensor(
    ...     data={
    ...         'edge_src': [1, 0, 3, 4, 2, 4, 3, 2],
    ...         'edge_dst': [0, 1, 2, 2, 3, 3, 4, 4],
    ...         'node_feature': [
    ...             [1.0, 0.0],
    ...             [1.0, 0.0],
    ...             [1.0, 0.0],
    ...             [1.0, 0.0],
    ...             [0.0, 1.0]
    ...         ],
    ...         'edge_feature': [
    ...             [1.0, 0.0],
    ...             [0.0, 1.0],
    ...             [0.0, 1.0],
    ...             [0.0, 1.0],
    ...             [1.0, 0.0],
    ...             [0.0, 1.0],
    ...             [1.0, 0.0],
    ...             [0.0, 1.0]
    ...         ],
    ...         'graph_indicator': [0, 0, 1, 1, 1],
    ...     }
    ... )
    >>> model = tf.keras.Sequential([
    ...     tf.keras.layers.Input(type_spec=graph_tensor.unspecific_spec),
    ...     molgraph.layers.NodeDropout(rate=0.15),
    ...     molgraph.layers.EdgeDropout(rate=0.15)
    ... ])
    >>> model.output_shape
    (None, 2)

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


@keras.saving.register_keras_serializable(package='molgraph')
class EdgeDropout(NodeDropout):
    '''Randomly dropping edges from the graph.

    **Example:**

    >>> graph_tensor = molgraph.GraphTensor(
    ...     data={
    ...         'edge_src': [1, 0, 3, 4, 2, 4, 3, 2],
    ...         'edge_dst': [0, 1, 2, 2, 3, 3, 4, 4],
    ...         'node_feature': [
    ...             [1.0, 0.0],
    ...             [1.0, 0.0],
    ...             [1.0, 0.0],
    ...             [1.0, 0.0],
    ...             [0.0, 1.0]
    ...         ],
    ...         'edge_feature': [
    ...             [1.0, 0.0],
    ...             [0.0, 1.0],
    ...             [0.0, 1.0],
    ...             [0.0, 1.0],
    ...             [1.0, 0.0],
    ...             [0.0, 1.0],
    ...             [1.0, 0.0],
    ...             [0.0, 1.0]
    ...         ],
    ...         'graph_indicator': [0, 0, 1, 1, 1],
    ...     }
    ... )
    >>> model = tf.keras.Sequential([
    ...     tf.keras.layers.Input(type_spec=graph_tensor.unspecific_spec),
    ...     molgraph.layers.NodeDropout(rate=0.15),
    ...     molgraph.layers.EdgeDropout(rate=0.15)
    ... ])
    >>> model.output_shape
    (None, 2)

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

