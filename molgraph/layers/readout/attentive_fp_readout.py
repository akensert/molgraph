import tensorflow as tf
from tensorflow import keras

from typing import Optional
from typing import Tuple
from typing import TypeVar

from molgraph.internal import register_keras_serializable 

from molgraph.layers.attentional.gat_conv import GATConv

from molgraph.tensors.graph_tensor import GraphTensor


@register_keras_serializable(package='molgraph')
class AttentiveFPReadout(tf.keras.layers.Layer):

    '''Readout step ("Molecule embedding") of AttentiveFP.

    For a complete implementation of AttentiveFP, add a number of `AttentiveFPConv` 
    steps before `AttentiveFPReadout` (see below).

    The implementation is based on Xiong et al. (2020) [#]_.

    Example usage:

    Create a complete AttentiveFP model:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     sizes=[2, 3],
    ...     node_feature=[[1., 0.], [1., 0.], [1., 0.], [1., 0.], [0., 1.]],
    ...     edge_feature=[[1., 0.], [0., 1.], [0., 1.], [0., 1.], 
    ...                   [1., 0.], [0., 1.], [1., 0.], [0., 1.]],
    ...     edge_src=[1, 0, 3, 4, 2, 4, 3, 2],
    ...     edge_dst=[0, 1, 2, 2, 3, 3, 4, 4],
    ... )
    >>> # Build a model with AttentiveFPConv
    >>> attentive_fp = tf.keras.Sequential([
    ...     molgraph.layers.AttentiveFPConv(16, apply_initial_node_projection=True),
    ...     molgraph.layers.AttentiveFPConv(16),
    ...     # ... 
    ...     molgraph.layers.AttentiveFPReadout(steps=4),
    ... ])
    >>> attentive_fp(graph_tensor).shape
    TensorShape([2, 16])

    Args:
        steps (int):
            Number of aggregation steps to perform. Default to 4.
        message_step (molgraph.layers.attentional.gat_conv.GATConv, None):
            The message passing step.
        update_step (tf.keras.layers.GRUCell, None):
            The update step. 
        final_node_projection (tf.keras.layers.Dense, None):
            The final projection applied to the output.

    References:
        .. [#] https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959
    '''

    def __init__(
        self,
        steps: int = 4,
        message_step: Optional[GATConv] = None,
        update_step: Optional[tf.keras.layers.GRUCell] = None,
        final_node_projection: Optional[tf.keras.layers.Dense] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.steps = steps
        self.message_step = message_step
        if self.message_step is None:
            self.message_step = GATConv(units=None, use_edge_features=False)
        self.update_step = update_step
        self.final_node_projection = final_node_projection

    def build(self, input_shape):
        super().build(input_shape)
        if self.update_step is None:
            self.update_step = tf.keras.layers.GRUCell(input_shape[-1])
        if self.final_node_projection is None:
            self.final_node_projection = tf.keras.layers.Dense(input_shape[-1])
    
    def call(self, tensor: GraphTensor) -> GraphTensor:

        tensor, virtual_node_indices = _add_virtual_super_nodes(tensor)

        virtual_node_state = tf.gather(
            tensor.node_feature, virtual_node_indices)
        
        for _ in range(self.steps):
            # Perform a step of GATConv (message passing)
            tensor = self.message_step(tensor)
            # Perform a step of GRU (message update)
            virtual_node_feature = tf.gather(
                tensor.node_feature, virtual_node_indices)
            virtual_node_state, _ = self.update_step(
                inputs=virtual_node_feature,
                states=virtual_node_state)

            node_feature_updated = tf.tensor_scatter_nd_update(
                tensor=tensor.node_feature, 
                indices=virtual_node_indices[:, tf.newaxis], 
                updates=virtual_node_state)
            tensor = tensor.update({'node_feature': node_feature_updated})
        
        return self.final_node_projection(virtual_node_state)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'steps': self.steps, 
            'message_step': tf.keras.layers.serialize(self.message_step),
            'update_step': tf.keras.layers.serialize(self.update_step),
            'final_node_projection': tf.keras.layers.serialize(
                self.final_node_projection)
        })
        return config


def _add_virtual_super_nodes(
    tensor: GraphTensor
) -> Tuple[GraphTensor, tf.Tensor]:
    
    def _get_edges(
        node_feature: tf.RaggedTensor
    ) -> Tuple[tf.RaggedTensor, tf.RaggedTensor]:
        graph_sizes = node_feature.row_lengths()
        edge_src_flat = tf.cast(
            tf.ragged.range(1, graph_sizes+1).flat_values, 
            dtype=tensor.edge_src.dtype)
        edge_dst_flat = tf.zeros(
            shape=(tf.reduce_sum(graph_sizes),), 
            dtype=tensor.edge_dst.dtype)
        edge_src = tf.RaggedTensor.from_row_lengths(
            edge_src_flat, graph_sizes)
        edge_dst = tf.RaggedTensor.from_row_lengths(
            edge_dst_flat, graph_sizes)
        return edge_src, edge_dst
    
    if isinstance(tensor.node_feature, tf.Tensor):
        # (None, node_dim) -> (batch_size, None, node_dim)
        node_feature = tf.RaggedTensor.from_value_rowids(
            tensor.node_feature, tensor.graph_indicator)
    else:
        node_feature = tensor.node_feature
    
    edge_src, edge_dst = _get_edges(node_feature)

    # (batch_size, None, node_dim) -> (batch_size, 1, node_dim)
    virtual_node_state = tf.reduce_sum(
        node_feature, axis=1, keepdims=True)
    
    # -> (batch_size, 1 + None, node_dim)
    node_feature = tf.concat([
        virtual_node_state, node_feature], axis=1)
    
    tensor = tensor.__class__(
        node_feature=node_feature,
        edge_src=edge_src,
        edge_dst=edge_dst)
    
    virtual_node_indices = tensor.node_feature.row_starts()
    
    tensor = tensor.merge()
    
    return tensor, virtual_node_indices