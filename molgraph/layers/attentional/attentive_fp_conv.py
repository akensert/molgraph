import tensorflow as tf
from tensorflow import keras

from keras import initializers
from keras import regularizers
from keras import constraints


from typing import Optional
from typing import Callable
from typing import Union

from molgraph.internal import register_keras_serializable 

from molgraph.tensors.graph_tensor import GraphTensor

from molgraph.layers.attentional.gat_conv import GATConv


@register_keras_serializable(package='molgraph')
class AttentiveFPConv(GATConv):

    '''Node message passing step ("Atom embedding") of AttentiveFP.

    `AttentiveFPConv` inherits from `GATConv` and adds a GRU update. Also
    performs a initial linear transformation on node features if needed or desired.

    For a complete implementation of AttentiveFP, add `AttentiveFPReadout` after 
    `AttentiveFPConv` steps.

    The implementation is based on Xiong et al. (2020) [#]_.

    Example usage:
    
    >>> graph_tensor = molgraph.GraphTensor(
    ...     sizes=[2, 3],
    ...     node_feature=[[1., 0.], [1., 0.], [1., 0.], [1., 0.], [0., 1.]],
    ...     edge_feature=[[1., 0.], [0., 1.], [0., 1.], [0., 1.], 
    ...                   [1., 0.], [0., 1.], [1., 0.], [0., 1.]],
    ...     edge_src=[1, 0, 3, 4, 2, 4, 3, 2],
    ...     edge_dst=[0, 1, 2, 2, 3, 3, 4, 4],
    ... )
    >>> # Build a model with AttentiveFPConv
    >>> gnn_model = tf.keras.Sequential([
    ...     molgraph.layers.AttentiveFPConv(16, apply_initial_node_projection=True),
    ...     molgraph.layers.AttentiveFPConv(16)
    ... ])
    >>> gnn_model(graph_tensor).node_feature.shape
    TensorShape([5, 16])

    Create a complete AttentiveFP model:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     sizes=[2, 3],
    ...     node_feature=[[1., 0.], [1., 0.], [1., 0.], [1., 0.], [0., 1.]],
    ...     edge_feature=[[1., 0.], [0., 1.], [0., 1.], [0., 1.], 
    ...                   [1., 0.], [0., 1.], [1., 0.], [0., 1.]],
    ...     edge_src=[1, 0, 3, 4, 2, 4, 3, 2],
    ...     edge_dst=[0, 1, 2, 2, 3, 3, 4, 4],
    ... )
    >>> attentive_fp = tf.keras.Sequential([
    ...     molgraph.layers.AttentiveFPConv(16, apply_initial_node_projection=True),
    ...     molgraph.layers.AttentiveFPConv(16),
    ...     molgraph.layers.AttentiveFPReadout(),
    ... ])
    >>> attentive_fp(graph_tensor).shape
    TensorShape([2, 16])

    Pass the same GRU cell to each layer to perform weight sharing:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     sizes=[2, 3],
    ...     node_feature=[[1., 0.], [1., 0.], [1., 0.], [1., 0.], [0., 1.]],
    ...     edge_feature=[[1., 0.], [0., 1.], [0., 1.], [0., 1.], 
    ...                   [1., 0.], [0., 1.], [1., 0.], [0., 1.]],
    ...     edge_src=[1, 0, 3, 4, 2, 4, 3, 2],
    ...     edge_dst=[0, 1, 2, 2, 3, 3, 4, 4],
    ... )
    >>> gru_cell = tf.keras.layers.GRUCell(16) # same units as AttentiveFPConv
    >>> gnn_model = tf.keras.Sequential([
    ...     molgraph.layers.AttentiveFPConv(
    ...         16, apply_initial_node_projection=True, gru_cell=gru_cell),
    ...     molgraph.layers.AttentiveFPConv(16, gru_cell=gru_cell)
    ... ])
    >>> gnn_model(graph_tensor).node_feature.shape
    TensorShape([5, 16])

    Args:
        units (int, None):
            Number of output units.
        apply_initial_node_projection (bool):
            Whether to perform an initial linear transformation on node features.
            Should be set to True for the first AttentiveFPConv layer in a sequence
            of AttentiveFPConv layers. Note that a node projection automatically 
            occurs when units != node_feature_shape[-1]. Default to False.
        gru_cell (tf.keras.layers.GRUCell, None):
            For weight-tying of the update step (GRU step) provide a GRU cell.
            Default to None.
        num_heads (int):
            Number of attention heads. Default to 8.
        merge_mode (str):
            The strategy for merging the heads. Either of 'concat', 'sum',
            'mean' or None. If set to None, 'mean' is used. Default to 'concat'.
        self_projection (bool):
            Whether to apply self projection. Default to True.
        normalization: (None, str, bool):
            Whether to apply layer normalization to the output. If batch 
            normalization is desired, pass 'batch_norm'. Default to None.
        residual: (bool)
            Whether to add skip connection to the output. Default to True.
        dropout: (float, None):
            Dropout applied to the output of the layer. Default to None.
        attention_activation (tf.keras.activations.Activation, callable, str, None):
            Activation function applied to the the attention scores.
            Default to 'leaky_relu'.
        activation (tf.keras.activations.Activation, callable, str, None):
            Activation function applied to the output of the layer.
            Default to 'relu'.
        use_bias (bool):
            Whether the layer should use biases. Default to True.
        kernel_initializer (tf.keras.initializers.Initializer, str):
            Initializer function for the kernels. Default to
            tf.keras.initializers.TruncatedNormal(stddev=0.005).
        bias_initializer (tf.keras.initializers.Initializer, str):
            Initializer function for the biases. Default to
            tf.keras.initializers.Constant(0.).
        kernel_regularizer (tf.keras.regularizers.Regularizer, None):
            Regularizer function applied to the kernels. Default to None.
        bias_regularizer (tf.keras.regularizers.Regularizer, None):
            Regularizer function applied to the biases. Default to None.
        activity_regularizer (tf.keras.regularizers.Regularizer, None):
            Regularizer function applied to the final output of the layer.
            Default to None.
        kernel_constraint (tf.keras.constraints.Constraint, None):
            Constraint function applied to the kernels. Default to None.
        bias_constraint (tf.keras.constraints.Constraint, None):
            Constraint function applied to the biases. Default to None.
        **kwargs: Valid (optional) keyword arguments are:
        
            *   `name` (str): Name of the layer instance.
            *   `update_step` (tf.keras.layers.Layer): Applies post-processing 
                step on the output (produced by `_call`). If passed, 
                `normalization`, `residual`, `activation` and `dropout` 
                parameters will be ignored. If None, a default post-processing 
                step will be used (taking into consideration the aforementioned 
                parameters). Default to None. 

    References:
        .. [#] https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959
    '''

    def __init__(
        self,
        units: Optional[int] = None,
        apply_initial_node_projection: bool = False,
        gru_cell: Optional[tf.keras.layers.GRUCell] = None,
        num_heads: int = 8,
        merge_mode: Optional[str] = 'concat',
        self_projection: bool = True,
        normalization: Union[None, str, bool] = None,
        residual: bool = True,
        dropout: Optional[float] = None,
        attention_activation: Union[
            None, str, Callable[[tf.Tensor], tf.Tensor]] = 'leaky_relu',
        activation: Union[None, str, Callable[[tf.Tensor], tf.Tensor]] = 'relu',
        use_bias: bool = True,
        kernel_initializer: Union[str, initializers.Initializer, None] = None,
        bias_initializer: Union[str, initializers.Initializer, None] = None,
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_regularizer: Optional[regularizers.Regularizer] = None,
        activity_regularizer: Optional[regularizers.Regularizer] = None,
        kernel_constraint: Optional[constraints.Constraint] = None,
        bias_constraint: Optional[constraints.Constraint] = None,
        **kwargs
    ):
    
        super().__init__(
            units=units,
            use_edge_features=apply_initial_node_projection,
            update_edge_features=False,
            num_heads=num_heads,
            merge_mode=merge_mode,
            self_projection=self_projection,
            normalization=normalization,
            residual=residual,
            dropout=dropout,
            attention_activation=attention_activation,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )

        self.apply_initial_node_projection = apply_initial_node_projection
        self.gru_cell = gru_cell
        self._built = False

    def _build(self, graph_tensor_spec: GraphTensor.Spec) -> None:
        
        node_feature_shape = graph_tensor_spec.node_feature.shape
        node_dim = node_feature_shape[-1]
        
        if self.apply_initial_node_projection or self.units != node_dim:
            self.initial_projection = self.get_dense(self.units, 'leaky_relu')
            node_feature_shape = node_feature_shape[:-1] + [self.units]
        else:
            self.initial_projection = None

        # Build GATConv
        super()._build(graph_tensor_spec)

        if not isinstance(self.gru_cell, tf.keras.layers.GRUCell):
            self.gru_cell = tf.keras.layers.GRUCell(self.units)

    def _call(self, tensor: GraphTensor) -> GraphTensor:

        if self.initial_projection is not None:
            tensor = tensor.update({
                'node_feature': self.initial_projection(tensor.node_feature)
            })

        node_feature_state = tensor.node_feature

        # Call GATConv
        tensor = super()._call(tensor)
        
        node_feature_state, _ = self.gru_cell(
            inputs=tensor.node_feature,
            states=node_feature_state
        )
        return tensor.update({'node_feature': node_feature_state})
    
    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'apply_initial_node_projection': self.apply_initial_node_projection,
            'gru_cell': tf.keras.layers.serialize(self.gru_cell)
        })
        return config