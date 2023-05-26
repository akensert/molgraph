import tensorflow as tf
from tensorflow import keras
from keras.utils import tf_utils
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.keras import activations


from typing import Optional
from typing import Callable
from typing import Union
from typing import Tuple
from typing import Type
from typing import TypeVar

from molgraph.tensors.graph_tensor import GraphTensor
from molgraph.layers.base import BaseLayer
from molgraph.layers.ops import softmax_edge_weights
from molgraph.layers.ops import propagate_node_features
from molgraph.layers.ops import reduce_features
from molgraph.layers.attentional.gat_conv import GATConv


Config = TypeVar('Config', bound=dict)


@keras.utils.register_keras_serializable(package='molgraph')
class AttentiveFPConv(GATConv):

    '''Node message passing step ("Atom embedding") of AttentiveFP.

    `AttentiveFPConv` inherits from `GATConv` and adds a GRU update. Also
    performs a initial linear transformation on node features if needed or desired.

    For a complete implementation of AttentiveFP, add `AttentiveFPReadout` after 
    `AttentiveFPConv` steps.

    The implementation is based on Xiong et al. (2020) [#]_.

    **Examples:**

    Inputs a ``GraphTensor`` encoding (two) subgraphs:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     data={
    ...         'edge_dst': [[0, 1], [0, 0, 1, 1, 2, 2]],
    ...         'edge_src': [[1, 0], [1, 2, 0, 2, 1, 0]],
    ...         'node_feature': [
    ...             [[1.0, 0.0], [1.0, 0.0]],
    ...             [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    ...         ],
    ...         'edge_feature': [
    ...             [[1.0, 0.0], [0.0, 1.0]],
    ...             [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0],
    ...              [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    ...         ],
    ...     }
    ... )
    >>> # Build a model with AttentiveFPConv
    >>> gnn_model = tf.keras.Sequential([
    ...     tf.keras.Input(type_spec=graph_tensor.unspecific_spec),
    ...     molgraph.layers.AttentiveFPConv(16, apply_initial_node_projection=True),
    ...     molgraph.layers.AttentiveFPConv(16)
    ... ])
    >>> gnn_model.output_shape
    (None, None, 16)

    Inputs a ``GraphTensor`` encoding a single disjoint graph:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     data={
    ...         'edge_dst': [0, 1, 2, 2, 3, 3, 4, 4],
    ...         'edge_src': [1, 0, 3, 4, 2, 4, 3, 2],
    ...         'node_feature': [
    ...             [1.0, 0.0],
    ...             [1.0, 0.0],
    ...             [1.0, 0.0],
    ...             [1.0, 0.0],
    ...             [0.0, 1.0]
    ...         ],
    ...         'graph_indicator': [0, 0, 1, 1, 1],
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
    ...     }
    ... )
    >>> # Build a model with AttentiveFPConv
    >>> gnn_model = tf.keras.Sequential([
    ...     tf.keras.Input(type_spec=graph_tensor.unspecific_spec),
    ...     molgraph.layers.AttentiveFPConv(16, apply_initial_node_projection=True),
    ...     molgraph.layers.AttentiveFPConv(16)
    ... ])
    >>> gnn_model.output_shape
    (None, 16)

    Create a complete AttentiveFP model:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     data={
    ...         'edge_dst': [0, 1, 2, 2, 3, 3, 4, 4],
    ...         'edge_src': [1, 0, 3, 4, 2, 4, 3, 2],
    ...         'node_feature': [
    ...             [1.0, 0.0],
    ...             [1.0, 0.0],
    ...             [1.0, 0.0],
    ...             [1.0, 0.0],
    ...             [0.0, 1.0]
    ...         ],
    ...         'graph_indicator': [0, 0, 1, 1, 1],
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
    ...     }
    ... )
    >>> # Build a model with AttentiveFPConv
    >>> attentive_fp = tf.keras.Sequential([
    ...     tf.keras.Input(type_spec=graph_tensor.unspecific_spec),
    ...     molgraph.layers.AttentiveFPConv(16, apply_initial_node_projection=True),
    ...     molgraph.layers.AttentiveFPConv(16),
    ...     molgraph.layers.AttentiveFPReadout(),
    ... ])
    >>> attentive_fp(graph_tensor).shape.as_list()
    [2, 16]


    Pass the same GRU cell to each layer to perform weight sharing:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     data={
    ...         'edge_dst': [0, 1, 2, 2, 3, 3, 4, 4],
    ...         'edge_src': [1, 0, 3, 4, 2, 4, 3, 2],
    ...         'node_feature': [
    ...             [1.0, 0.0],
    ...             [1.0, 0.0],
    ...             [1.0, 0.0],
    ...             [1.0, 0.0],
    ...             [0.0, 1.0]
    ...         ],
    ...         'graph_indicator': [0, 0, 1, 1, 1],
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
    ...     }
    ... )
    >>> gru_cell = tf.keras.layers.GRUCell(16) # same units as AttentiveFPConv
    >>> # Build a model with AttentiveFPConv
    >>> gnn_model = tf.keras.Sequential([
    ...     tf.keras.Input(type_spec=graph_tensor.unspecific_spec),
    ...     molgraph.layers.AttentiveFPConv(
    ...         16, apply_initial_node_projection=True, gru_cell=gru_cell),
    ...     molgraph.layers.AttentiveFPConv(16, gru_cell=gru_cell)
    ... ])
    >>> gnn_model.output_shape
    (None, 16)


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
        batch_norm: (bool):
            Whether to apply batch normalization to the output. Default to True.
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
        batch_norm: bool = True,
        residual: bool = True,
        dropout: Optional[float] = None,
        attention_activation: Union[
            None, str, Callable[[tf.Tensor], tf.Tensor]] = 'leaky_relu',
        activation: Union[None, str, Callable[[tf.Tensor], tf.Tensor]] = 'relu',
        use_bias: bool = True,
        kernel_initializer: Union[
            str, initializers.Initializer
        ] = initializers.TruncatedNormal(stddev=0.005),
        bias_initializer: Union[
            str, initializers.Initializer
        ] = initializers.Constant(0.),
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
            num_heads=num_heads,
            merge_mode=merge_mode,
            self_projection=self_projection,
            batch_norm=batch_norm,
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

    def subclass_build(self, node_feature_shape, edge_feature_shape):

        if self.apply_initial_node_projection or self.units != node_feature_shape[-1]:
            self.initial_projection = self.get_dense(self.units, 'leaky_relu')
            node_feature_shape = node_feature_shape[:-1] + [self.units]
        else:
            self.initial_projection = None

        super().subclass_build(node_feature_shape, edge_feature_shape)

        if not isinstance(self.gru_cell, tf.keras.layers.GRUCell):
            self.gru_cell = tf.keras.layers.GRUCell(self.units)

    def subclass_call(self, tensor: GraphTensor) -> GraphTensor:

        if self.initial_projection is not None:
            tensor = tensor.update({
                'node_feature': self.initial_projection(
                    tensor.node_feature)
            })

        node_feature_state = tensor.node_feature

        tensor = super().subclass_call(tensor)
        
        node_feature_state, _ = self.gru_cell(
            inputs=tensor.node_feature,
            states=node_feature_state
        )
        return tensor.update({'node_feature': node_feature_state})
    
    def get_config(self) -> Config:
        config = super().get_config()
        config.update({
            'apply_initial_node_projection': self.apply_initial_node_projection,
            'gru_cell': tf.keras.layers.serialize(self.gru_cell)
        })
        return config