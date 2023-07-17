import tensorflow as tf
from tensorflow import keras

from keras import initializers
from keras import regularizers
from keras import constraints
from keras import activations

from typing import Optional
from typing import Callable
from typing import Union

from molgraph.internal import register_keras_serializable 

from molgraph.tensors.graph_tensor import GraphTensor
from molgraph.tensors.graph_tensor import GraphTensorSpec

from molgraph.layers import gnn_layer


@register_keras_serializable(package='molgraph')
class GatedGCNConv(gnn_layer.GNNLayer):

    '''Gated graph convolutional layer (GatedGCN).

    Implementation is based on Dwivedi et al. (2022) [#]_, Bresson et al. (2019) [#]_,
    Joshi et al. (2019) [#]_, and Bresson et al. (2018) [#]_.

    **Examples:**

    Inputs a ``GraphTensor`` encoding (two) subgraphs:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     data={
    ...         'edge_src': [[1, 0], [1, 2, 0, 2, 1, 0]],
    ...         'edge_dst': [[0, 1], [0, 0, 1, 1, 2, 2]],
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
    >>> # Build a model with GatedGCNConv
    >>> gnn_model = tf.keras.Sequential([
    ...     tf.keras.Input(type_spec=graph_tensor.unspecific_spec),
    ...     molgraph.layers.GatedGCNConv(16, activation='relu'),
    ...     molgraph.layers.GatedGCNConv(16, activation='relu')
    ... ])
    >>> gnn_model.output_shape
    (None, None, 16)

    Inputs a ``GraphTensor`` encoding a single disjoint graph:

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
    >>> # Build a model with GatedGCNConv
    >>> gnn_model = tf.keras.Sequential([
    ...     tf.keras.Input(type_spec=graph_tensor.unspecific_spec),
    ...     molgraph.layers.GatedGCNConv(16, activation='relu'),
    ...     molgraph.layers.GatedGCNConv(16, activation='relu')
    ... ])
    >>> gnn_model.output_shape
    (None, 16)

    Args:
        units (int, None):
            Number of output units.
        use_edge_features (bool):
            Whether or not to use edge features. Default to True.
        self_projection (bool):
            Whether to apply self projection. Default to True.
        normalization: (None, str, bool):
            Whether to apply layer normalization to the output. If batch 
            normalization is desired, pass 'batch_norm'. Default to True.
        residual: (bool)
            Whether to add skip connection to the output. Default to True.
        dropout: (float, None):
            Dropout applied to the output of the layer. Default to None.
        activation (tf.keras.activations.Activation, callable, str, None):
            Activation function applied to the output of the layer. Default to 'relu'.
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
            *   `use_edge_features`: Whether or not to use edge features. 
                Only relevant if edge features exist. If None, and edge 
                features exist, it will be set to True. Default to None.
            *   `update_edge_features` (bool): Specifies whether edge features 
                should be updated along with node features, including the 
                post-processing step. Only relevant if edge features exist. 
                It is important that GNN layers which updates its edge features
                for the next layer sets this to True. Default to False. 

    References:
        .. [#] https://arxiv.org/pdf/2003.00982.pdf
        .. [#] https://arxiv.org/pdf/1906.03412.pdf
        .. [#] https://arxiv.org/pdf/1906.01227.pdf
        .. [#] https://arxiv.org/pdf/1711.07553.pdf

    '''

    def __init__(
        self,
        units: Optional[int] = None,
        self_projection: bool = True,
        normalization: Union[None, str, bool] = 'layer_norm',
        residual: bool = True,
        dropout: Optional[float] = None,
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
        kwargs['update_edge_features'] = (
            kwargs.get('update_edge_features', True) and 
            kwargs.get('use_edge_features', True)
        )
        super().__init__(
            units=units,
            normalization=normalization,
            residual=residual,
            dropout=dropout,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

        self.gate_activation = activations.get('sigmoid')
        self.apply_self_projection = self_projection

    def _build(self, graph_tensor_spec: GraphTensorSpec) -> None:

        if self.use_edge_features:
            self.edge_gate_projection = self.get_dense(self.units)

        self.node_src_gate_projection = self.get_dense(self.units)
        self.node_dst_gate_projection = self.get_dense(self.units)

        self.node_projection = self.get_dense(self.units)
        if self.apply_self_projection:
            self.self_projection = self.get_dense(self.units)

    def _call(self, tensor: GraphTensor) -> GraphTensor:

        if self.apply_self_projection:
            node_feature_residual = self.self_projection(tensor.node_feature)
        else:
            node_feature_residual = None

        # Edge dependent (`tensor.edge_src is not None`), from here
        node_feature = self.node_projection(tensor.node_feature)

        node_feature_src = tf.gather(
            tensor.node_feature, tensor.edge_src)
        node_feature_dst = tf.gather(
            tensor.node_feature, tensor.edge_dst)

        gate = self.node_src_gate_projection(node_feature_src)
        gate += self.node_dst_gate_projection(node_feature_dst)

        if self.use_edge_features:
            edge_feature = self.edge_gate_projection(tensor.edge_feature)
            gate += edge_feature
            tensor = tensor.update({'edge_feature': gate})

        gate = self.gate_activation(gate)

        tensor = tensor.update({
            'node_feature': node_feature, 'edge_weight': gate})
        
        return tensor.propagate(
            normalize=True, 
            residual=node_feature_residual,
            exponentiate=False)

    def get_config(self):
        base_config = super().get_config()
        config = {
            'self_projection': self.apply_self_projection,
        }
        base_config.update(config)
        return base_config
