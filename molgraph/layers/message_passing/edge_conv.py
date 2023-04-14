import tensorflow as tf
from tensorflow import keras
from keras.utils import tf_utils
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.keras import activations
from tensorflow.keras import layers
import math

from typing import Optional
from typing import Callable
from typing import Union
from typing import Tuple
from typing import Type
from typing import TypeVar

from molgraph.tensors.graph_tensor import GraphTensor
from molgraph.layers.base import BaseLayer
from molgraph.layers.ops import compute_edge_weights_from_degrees
from molgraph.layers.ops import propagate_node_features


Config = TypeVar('Config', bound=dict)


@keras.utils.register_keras_serializable(package='molgraph')
class EdgeConv(keras.layers.Layer):

    """Edge convolutional layer, used to build DMPNN [#]_ and DGIN [#_] like models.

    **Important:**

    As of now, EdgeConv only works on (sub)graphs with at least one edge/bond. If your dataset consists
    of molecules with a single atom, please add self loops: 
    ``molgraph.chemistry.MolecularGraphEncoder(..., self_loops=True)``

    **Examples:**

    Build a DGIN model. Note that the DGIN model below differs from `molgraph.models.DGIN` in some
    ways. For instance, it uses the very previous edge states (in `molgraph.layers.EdgeConv`) and 
    node states (in `molgraph.layers.GINConv`) as residual ("skip connections") rather than the very 
    initial edge states and node states respectively. Furthermore, this implementation also allows us to use
    GRU as the update function, though default is to use a Dense layer. Like the models
    (`molgraph.models.DGIN` and `molgraph.models.DMPNN`), this implementation allows for both shared 
    ("weight tying") and unshared weights. Simply pass the previous layer via the `tie_layer` argument.

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
    >>> # Build a DGIN model for binary classificaton
    >>> gnn_model = tf.keras.Sequential([
    ...     tf.keras.Input(type_spec=graph_tensor.unspecific_spec),
    ...     molgraph.layers.EdgeConv(16),                           # will produce 'edge_state' field
    ...     molgraph.layers.EdgeConv(16),                           # will produce 'edge_state' field
    ...     molgraph.layers.NodeReadout(target='edge_state'),       # target='edge_state' is default
    ...     molgraph.layers.GINConv(32),
    ...     molgraph.layers.GINConv(32),
    ...     molgraph.layers.Readout(),
    ...     tf.keras.layers.Dense(1, activation='sigmoid')
    ... ])
    >>> gnn_model.output_shape
    (None, 1)

    Tie weights of the previous layers with the subsequent layers:

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
    >>> # Build a model with EdgeConv
    >>> edge_conv_1 = molgraph.layers.EdgeConv(
    ...     units=16, 
    ...     update_mode='gru'       # lets use GRU updates instead (as per Gilmer et al. (2017))
    ... )
    >>> edge_conv_2 = molgraph.layers.EdgeConv(
    ...     units=32,               # will be ignored as EdgeConv layer is passed to `tie_layer`
    ...     update_mode='gru',      # will be ignored as EdgeConv layer is passed to `tie_layer`
    ...     tie_layer=edge_conv_1
    ... )
    >>> edge_conv_3 = molgraph.layers.EdgeConv(
    ...     units=16,               # will be ignored as EdgeConv layer is passed to `tie_layer`
    ...     update_mode='gru',      # will be ignored as EdgeConv layer is passed to `tie_layer`
    ...     tie_layer=edge_conv_1   # specifying edge_conv_2 would be equivalent
    ... )
    >>> gnn_model = tf.keras.Sequential([
    ...     tf.keras.Input(type_spec=graph_tensor.unspecific_spec),
    ...     edge_conv_1,
    ...     edge_conv_2,
    ...     edge_conv_3,
    ...     molgraph.layers.NodeReadout()
    ... ])
    >>> gnn_model.output_shape
    (None, 16)

    Args:
        units (int, None):
            Number of output units.
        update_mode (bool):
            Specify what type of update will be performed. Either 'dense' or 'gru'. 
            If 'gru' is specified, make sure weight tying is performed
            (see 'tie_layer' argument above). Default to 'dense'.
        tie_layer (molgraph.layers.message_passing.edge_conv.EdgeConv, None):
            Pass the previous EdgeConv layer to perform weight tying. If None, weight tying
            will not be performed (each layer has its own weights). Default to None.
        activation (tf.keras.activations.Activation, callable, str, None):
            Activation function applied to the updated edge states. If None is set, either 'relu'
            (for `update_mode='dense'`) or 'tanh' (for `update_mode='gru'`) will be used. 
            Default to None.
        recurrent_activation (tf.keras.activations.Activation, callable, str, None):
            Activation function applied to the recurrent step (only relevant if ``update_mode='gru'``).
            Default to to 'sigmoid'.
        use_bias (bool):
            Whether the layer should use biases. If None is set, either False or True will be used.
            Default to None.
        kernel_initializer (tf.keras.initializers.Initializer, str):
            Initializer function for the kernels. If None is set, either
            ``tf.keras.initializers.TruncatedNormal(stddev=0.005)`` (dense) or ``glorot_uniform`` (gru)
            will be used. Default to None.
        bias_initializer (tf.keras.initializers.Initializer, str):
            Initializer function for the biases. Default to
            tf.keras.initializers.Constant(0.).
        recurrent_initializer (tf.keras.initializers.Initializer, str):
            Initializer function for the recurrent kernel (only relevant if ``update_mode='gru'``). 
            Default to 'orthogonal'.
        kernel_regularizer (tf.keras.regularizers.Regularizer, None):
            Regularizer function applied to the kernels. Default to None.
        bias_regularizer (tf.keras.regularizers.Regularizer, None):
            Regularizer function applied to the biases. Default to None.
        activity_regularizer (tf.keras.regularizers.Regularizer, None):
            Regularizer function applied to the final output of the layer.
            Default to None.
        recurrent_regularizer (tf.keras.regularizers.Regularizer, None):
            Regularizer function applied to the recurrent kernel (only relevant if ``update_mode='gru'``).
            Default to None.
        kernel_constraint (tf.keras.constraints.Constraint, None):
            Constraint function applied to the kernels. Default to None.
        bias_constraint (tf.keras.constraints.Constraint, None):
            Constraint function applied to the biases. Default to None.
        recurrent_constraint (tf.keras.constraints.Constraint, None):
            Constraint function applied to the recurrent kernel (only relevant if ``update_mode='gru'``).
            Default to None.
        automatically_infer_gru_activation (bool):
            Whether to automatically infer the activation for GRU (when `activation=None`). Default to True.
        parallel_iterations (int, None):
            Number of ``parallel_iterations`` to be set for ``tf.map_fn`` to find
            the reverse edge states to be subtracted from the aggregated edge states.

    References:
        .. [#] https://arxiv.org/pdf/1904.01561.pdf
        .. [#] https://www.mdpi.com/1420-3049/26/20/6185
    """

    def __init__(
        self,
        units: Optional[int] = None,
        update_mode: str = 'dense',
        tie_layer: Optional['EdgeConv'] = None,
        activation: Union[str, None, Callable[[tf.Tensor], tf.Tensor]] = None,
        recurrent_activation: Union[str, None, Callable[[tf.Tensor], tf.Tensor]] = 'sigmoid',
        use_bias: Optional[bool] = None,
        kernel_initializer: Union[str, None, initializers.Initializer] = None,
        bias_initializer: Union[str, None, initializers.Initializer] = 'zeros',
        recurrent_initializer: Union[str, None, initializers.Initializer] = 'orthogonal',
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_regularizer: Optional[regularizers.Regularizer] = None,
        recurrent_regularizer: Optional[regularizers.Regularizer] = None,
        activity_regularizer: Optional[regularizers.Regularizer] = None,
        kernel_constraint: Optional[constraints.Constraint] = None,
        bias_constraint: Optional[constraints.Constraint] = None,
        recurrent_constraint: Optional[constraints.Constraint] = None,
        automatically_infer_gru_activation: bool = True,
        parallel_iterations: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.update_mode = update_mode
        self.tie_layer = tie_layer
        if not activation and update_mode != 'dense' and automatically_infer_gru_activation:
            activation = 'tanh'
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        if use_bias is None:
            use_bias = False if update_mode == 'dense' else True
        self.use_bias = use_bias
        if kernel_initializer is None:
            kernel_initializer = (
                initializers.TruncatedNormal(stddev=0.005) if update_mode == 'dense' 
                else 'glorot_uniform')
        self.kernel_initializer  = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer or 'zeros')
        self.recurrent_initializer = initializers.get(recurrent_initializer or 'orthogonal')
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self._parallel_iterations = parallel_iterations
        self._automatically_infer_gru_activation = automatically_infer_gru_activation
        self._initialize_edge_state = None
        self._built = False

    def _build(self, units, initialize_edge_state):

        if self.tie_layer:
            self.update_projection = self._get_tied_layer_projection()
            self.units = self.tie_layer.units
        else:
            if initialize_edge_state:
                self.initial_projection = self._get_dense(units)
            self.update_projection = (
                self._get_dense(units) if self.update_mode == 'dense' 
                else self._get_gru(units))

    def call(self, tensor: GraphTensor) -> GraphTensor:

        tensor_orig = tensor
        if isinstance(tensor.node_feature, tf.RaggedTensor):
            tensor = tensor.merge()

        # EdgeConv requires edge features, so if edge features do not exist,
        # we force edge features by initializing them as ones vector
        if not hasattr(tensor, 'edge_feature'):
            tensor = tensor.update({
                'edge_feature': tf.ones(
                    shape=[tf.shape(tensor.edge_dst)[0], 1], dtype=tf.float32)})

        if not self._built:
            with tf_utils.maybe_init_scope(self):
                self._initialize_edge_state = (
                    True if not hasattr(tensor, 'edge_state') else False)
                self.units = (
                    self.units if (self.units and (self._initialize_edge_state or self.update_mode == 'dense'))
                    else tensor.edge_state.shape[-1])
                self._build(self.units, self._initialize_edge_state)
                self._built = True

        if self._initialize_edge_state:
            edge_state = tf.gather(tensor.node_feature, tensor.edge_src)
            edge_state = tf.concat([edge_state, tensor.edge_feature], axis=-1)
            edge_state = self.initial_projection(edge_state)
            tensor = tensor.update({'edge_state': edge_state})
                
        edge_state_update = edge_message_step(
            edge_feature=tensor.edge_state,
            edge_src=tensor.edge_src,
            edge_dst=tensor.edge_dst,
            graph_indicator=tensor.graph_indicator,
            parallel_iterations=self._parallel_iterations)

        edge_state_update = edge_update_step(
            edge_feature=edge_state_update,
            edge_feature_prev=tensor.edge_state,
            update_projection=self.update_projection)
        
        return tensor_orig.update({'edge_state': edge_state_update})

    def get_config(self) -> Config:
        base_config = super().get_config()
        config = {
            'units': self.units,
            'update_mode': self.update_mode,
            'tie_layer': keras.layers.serialize(self.tie_layer),
            'activation':
                activations.serialize(self.activation),
            'recurrent_activation':
                activations.serialize(self.recurrent_activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer ),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'automatically_infer_gru_activation': self._automatically_infer_gru_activation,
            'parallel_iterations': self._parallel_iterations,
            'initialize_edge_state': self._initialize_edge_state,
        }
        base_config.update(config)
        return base_config

    @classmethod
    def from_config(cls: Type['EdgeConv'], config: Config) -> 'EdgeConv':
        initialize_edge_state = config.pop('initialize_edge_state')
        layer = cls(**config)
        layer._build(config['units'], initialize_edge_state)
        return layer
    
    def compute_output_shape(
        self,
        input_shape
    ) -> tf.TensorShape:
        return input_shape
    
    def _get_dense(self, units):
        return keras.layers.Dense(
            units=units, 
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer ,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint)
    
    def _get_gru(self, units):
        return keras.layers.GRUCell(
            units=units, 
            activation=self.activation,
            recurrent_activation=self.recurrent_activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer ,
            bias_initializer=self.bias_initializer,
            recurrent_initializer=self.recurrent_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            recurrent_regularizer=self.recurrent_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            recurrent_constraint=self.recurrent_constraint)

    def _get_tied_layer_projection(self):
        self._check_layer_types()
        update_projection = getattr(self.tie_layer, 'update_projection', None)
        if not update_projection:
            raise ValueError(
                f'`{self.tie_layer.name}` is not built. Make sure `{self.tie_layer.name}` is ' +
                f'built before building this layer (`{self.name}`). `{self.tie_layer.name}` should ' + 
                f'come before this layer (`{self.name}`) in the sequence (model).')   
        return update_projection

    def _check_layer_types(self):
        if type(self) != type(self.tie_layer):
            raise ValueError(
                f'`{self.tie_layer.name}` needs to be the same type as this layer (`{self.name}`)')

    
def edge_update_step(
    edge_feature: tf.Tensor, 
    edge_feature_prev: tf.Tensor,
    update_projection: Union[
        keras.layers.Dense, keras.layers.GRUCell, keras.layers.LSTMCell],
) -> tf.Tensor:
    if isinstance(update_projection, keras.layers.Dense):
        edge_feature_update = update_projection(
            tf.concat([edge_feature_prev, edge_feature], axis=1))
    else: # if keras.layers.GRUCell
        edge_feature_update, _ = update_projection(
            inputs=edge_feature,
            states=edge_feature_prev)
    return edge_feature_update

def edge_message_step(
    edge_feature: tf.Tensor, # or 'edge_state'
    edge_src: tf.Tensor,
    edge_dst: tf.Tensor,
    graph_indicator: tf.Tensor,
    parallel_iterations: Optional[int] = None
) -> tf.Tensor:
    num_nodes = tf.maximum(tf.reduce_max(edge_src), tf.reduce_max(edge_dst)) + 1
    message = tf.math.unsorted_segment_sum(edge_feature, edge_dst, num_nodes)
    message = tf.gather(message, edge_src)
    message -= _get_reverse_edge_features(
        edge_feature, edge_src, edge_dst, graph_indicator, parallel_iterations)
    return message

@tf.function
def _get_reverse_edge_features(
    edge_feature: tf.Tensor, 
    edge_src: tf.Tensor,
    edge_dst: tf.Tensor,
    graph_indicator: tf.Tensor,
    parallel_iterations: Optional[int] = None
) -> tf.Tensor:
    # Make tensors ragged to that they can be iterated over by tf.map_fn
    graph_indicator_edges = tf.gather(graph_indicator, edge_dst)
    edge_feature, edge_src, edge_dst = tf.nest.map_structure(
        lambda x: tf.RaggedTensor.from_value_rowids(x, graph_indicator_edges),
        (edge_feature, edge_src, edge_dst))
    # Define appropriate output spec
    output_spec = tf.RaggedTensorSpec(
        shape=[None, edge_feature.shape[-1]], 
        ragged_rank=0, 
        dtype=tf.float32)
    # Find all reverse edge features in the whole graph
    reverse_edge_state = tf.map_fn(
        fn=_get_reverse_edge_features_fn, 
        elems=(edge_feature, edge_src, edge_dst), 
        fn_output_signature=output_spec,
        parallel_iterations=parallel_iterations)
    # Convert ragged tensor output to a tensor.
    return reverse_edge_state.merge_dims(outer_axis=0, inner_axis=1)

def _get_reverse_edge_features_fn(
    inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor], 
) -> tf.Tensor:
    '''This function finds the reverse edge features/states for each molecule/subgraph.

    Will be called by `tf.map_fn` in `_get_reverse_edge_features`.
    '''
    edge_feature, edge_src, edge_dst = inputs
    # Find the index of "reverse" edge of "forward" edge edge_src->edge_dst
    edge_exclude = tf.logical_and(
        edge_src[:, None] == edge_dst,
        edge_dst[:, None] == edge_src)
    # Obtain index of edge_src->edge_dst ("forward") and its corresponding
    # edge_src<-edge_dst ("reverse"). For molecules: forward and reverse
    # edges are usually the same and always exist. 
    edge_forward, edge_reverse = tf.split(tf.where(edge_exclude), 2, axis=-1)
    return tf.tensor_scatter_nd_add(
        tf.zeros_like(edge_feature), 
        tf.expand_dims(edge_forward, -1), 
        tf.gather(edge_feature, edge_reverse))

