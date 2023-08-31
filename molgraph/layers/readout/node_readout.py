import tensorflow as tf
from tensorflow import keras

from typing import Optional

from molgraph.internal import register_keras_serializable 

from molgraph.tensors.graph_tensor import GraphTensor


@register_keras_serializable(package='molgraph')
class NodeReadout(keras.layers.Layer):

    '''Aggregates edge states to associated nodes.

    Example usage:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     sizes=[2, 3],
    ...     node_feature=[[1., 0.], [1., 0.], [1., 0.], [1., 0.], [0., 1.]],
    ...     edge_feature=[[1., 0.], [0., 1.], [0., 1.], [0., 1.], 
    ...                   [1., 0.], [0., 1.], [1., 0.], [0., 1.]],
    ...     edge_src=[1, 0, 3, 4, 2, 4, 3, 2],
    ...     edge_dst=[0, 1, 2, 2, 3, 3, 4, 4],
    ... )
    >>> gnn_model = tf.keras.Sequential([
    ...     molgraph.layers.EdgeConv(16),                           # will produce 'edge_state' component
    ...     molgraph.layers.EdgeConv(16),                           # will produce 'edge_state' component
    ...     molgraph.layers.NodeReadout(target='edge_state'),       # target='edge_state' is default
    ... ])
    >>> gnn_model(graph_tensor).node_feature.shape
    TensorShape([5, 16])

    Args:
        target (str):
            Specifies which component to aggregate. Default to 'edge_state' which is the
            component produced by ``molgraph.layers.EdgeConv``.
        apply_transform (bool):
            Whether to perform a transformaton after the aggregation. Default to False.
        dense_kwargs (None, dict):
            Parameters to be passed to the dense layer in the transformation. Only relevant
            if ``apply_transform=True``. If None is passed, ``units`` is set to ``input_shape[-1]``
            and ``activation`` is set to ``relu``. Default to None.
    '''

    def __init__(
        self, 
        target: str = 'edge_state', 
        apply_transform: bool = True,
        dense_kwargs: Optional[dict] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.target = target 
        self.apply_transform = apply_transform
        self.dense_kwargs = {} if dense_kwargs is None else dense_kwargs
        self._target_shape = None
        self._built = False

    def _build(self, input_shape: tf.TensorShape) -> None:
        with tf.init_scope():
            self._target_shape = input_shape
            if self.apply_transform:
                # If units/activation is set to None, use default (input_shape[-1]/'relu')
                self.dense_kwargs['units'] = self.dense_kwargs.pop(
                    'units', input_shape[-1])
                self.dense_kwargs['activation'] = self.dense_kwargs.pop(
                    'activation', 'relu')
                self.projection = keras.layers.Dense(**self.dense_kwargs)
            self._built = True

    def call(self, tensor: GraphTensor) -> GraphTensor:
        tensor_orig = tensor
        if isinstance(tensor.node_feature, tf.RaggedTensor):
            tensor = tensor.merge()
        if not self._built:
            self._build(getattr(tensor, self.target).shape)
        node_state = tf.math.unsorted_segment_sum(
            getattr(tensor, self.target), 
            tensor.edge_dst, 
            tf.shape(tensor.node_feature)[0])
        node_state = tf.concat([tensor.node_feature, node_state], axis=-1)
        if self.apply_transform:
            node_state = self.projection(node_state)
        return tensor_orig.update({'node_feature': node_state})
    
    def from_config(cls, config):
        target_shape = config.pop('target_shape')
        layer = cls(**config)
        layer._build(target_shape)
        return layer

    def get_config(self):
        config = super().get_config()
        config.update({
            'target': self.target,
            'apply_transform': self.apply_transform,
            'dense_kwargs': self.dense_kwargs,
            'target_shape': self._target_shape})
        return config
    
    def compute_output_shape(self, input_shape):
        return input_shape