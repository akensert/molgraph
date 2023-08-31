import tensorflow as tf
from tensorflow import keras

from keras import initializers
from keras import regularizers
from keras import constraints
from keras import activations

from typing import Optional
from typing import Callable
from typing import Union
from typing import Type

from molgraph.internal import register_keras_serializable 

from molgraph.tensors.graph_tensor import GraphTensor


@register_keras_serializable(package='molgraph')
class EdgeConv(tf.keras.layers.Layer):

    '''Edge convolutional layer, used to build DMPNN [#]_ and DGIN [#]_ like models.

    **Important:**

    As of now, EdgeConv only works on (sub)graphs with at least one edge/bond. If your dataset consists
    of molecules with a single atom, please add self loops: 
    ``molgraph.chemistry.MolecularGraphEncoder(..., self_loops=True)``

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
    ...     molgraph.layers.EdgeConv(16),                      # will produce 'edge_state' component
    ...     molgraph.layers.EdgeConv(16),                      # will produce 'edge_state' component
    ...     molgraph.layers.NodeReadout(target='edge_state'),  # target='edge_state' is default
    ...     molgraph.layers.Readout(),
    ... ])
    >>> gnn_model(graph_tensor).shape
    TensorShape([2, 16])

    Args:
        units (int, None):
            Number of output units.
        update_mode (bool):
            Specify what type of update will be performed. Either 'dense' or 'gru'. 
            Default to 'gru'.
        update_fn (tf.keras.layers.GRUCell, tf.keras.layers.Dense, None):
            Optionally pass update function (GRUCell or Dense) for weight-tying 
            of the update step (GRU step or Dense step). Default to None.
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

    References:
        .. [#] https://arxiv.org/pdf/1904.01561.pdf
        .. [#] https://www.mdpi.com/1420-3049/26/20/6185
    '''
    
    def __init__(
        self,
        units: Optional[int],
        update_mode: str = 'dense',
        update_fn: Optional[Union[
            tf.keras.layers.GRUCell, tf.keras.layers.Dense]] = None,
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
        **kwargs
    ):
        self_projection = kwargs.pop('self_projection', False)

        if self_projection:
            raise ValueError('`EdgeConv` does not support self projection.')
        
        super().__init__(**kwargs)

        if units is None:
            raise ValueError('`EdgeConv` requires units (int) to be passed.')
        
        self.units = units
        self.update_fn = update_fn
        self.update_mode = update_mode.lower()
        if activation is None:
            if self.update_mode.startswith('gru'):
                self.activation = 'tanh'
            else:
                self.activation = 'linear'
        else:
            self.activation = activation
        if use_bias is None:
            use_bias = True if update_mode.startswith('gru') else True
        self.use_bias = use_bias
        self.recurrent_activation = activations.get(recurrent_activation)
        if kernel_initializer is None:
            kernel_initializer = (
                'glorot_uniform' if self.update_mode.startswith('gru') 
                else initializers.TruncatedNormal(stddev=0.005)
            )
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
        self._initialize_edge_state = None
        self._built = False

    def _build(self, initialize_edge_state):
        
        with tf.init_scope():

            if initialize_edge_state:
                self._initialize_edge_state = True
                self.initial_projection = self._get_dense(self.units)
            else:
                self._initialize_edge_state = False

            if self.update_fn is None:
                self.update_fn = (
                    self._get_gru(self.units) if self.update_mode.startswith('gru')
                    else self._get_dense(self.units)
                )
            elif self.units != self.update_fn.units:
                raise ValueError(
                    'units of `update_fn` needs to match units of this layer.')

            self._built = True

    def call(self, tensor: GraphTensor) -> GraphTensor:

        tensor_orig = tensor
        if isinstance(tensor.node_feature, tf.RaggedTensor):
            tensor = tensor.merge()

        # EdgeConv requires edge features, so if edge features do not exist,
        # we force edge features by initializing them as ones vector
        if tensor.edge_feature is None:
            tensor = tensor.update({
                'edge_feature': tf.ones(
                    shape=[tf.shape(tensor.edge_src)[0], 1], dtype=tf.float32)})
            
        if not self._built:
            initialize_edge_state = (
                True if not hasattr(tensor, 'edge_state') else False)
            self._build(initialize_edge_state)

        if self._initialize_edge_state:
            edge_state = tf.gather(tensor.node_feature, tensor.edge_src)
            edge_state = tf.concat([edge_state, tensor.edge_feature], axis=-1)
            edge_state = self.initial_projection(edge_state)
            tensor = tensor.update({'edge_state': edge_state})
        
        edge_state_update = edge_message_step(
            edge_feature=tensor.edge_state,
            edge_src=tensor.edge_src,
            edge_dst=tensor.edge_dst)

        edge_state_update = edge_update_step(
            edge_feature=edge_state_update,
            edge_feature_prev=tensor.edge_state,
            update_projection=self.update_fn)
        
        return tensor_orig.update({'edge_state': edge_state_update})

    @classmethod
    def from_config(cls: Type['EdgeConv'], config: dict) -> 'EdgeConv':
        initialize_edge_state = config.pop('initialize_edge_state') 
        layer = cls(**config)
        if initialize_edge_state is None:
            pass
        else:
            layer._build(initialize_edge_state)
        return layer
    
    def get_config(self) -> dict:
        base_config = super().get_config()
        config = {
            'units': self.units,
            'update_mode': self.update_mode,
            'update_fn': tf.keras.layers.serialize(self.update_fn),
            'activation':
                activations.serialize(self.activation),
            'recurrent_activation':
                activations.serialize(self.recurrent_activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
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
            'initialize_edge_state': self._initialize_edge_state,
        }
        base_config.update(config)
        return base_config
    
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
            kernel_initializer=self.kernel_initializer.from_config(
                self.kernel_initializer.get_config()),
            bias_initializer=self.bias_initializer.from_config(
                self.bias_initializer.get_config()),
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
            kernel_initializer=self.kernel_initializer.from_config(
                self.kernel_initializer.get_config()),
            bias_initializer=self.bias_initializer.from_config(
                self.bias_initializer.get_config()),
            recurrent_initializer=self.recurrent_initializer.from_config(
                self.recurrent_initializer.get_config()),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            recurrent_regularizer=self.recurrent_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            recurrent_constraint=self.recurrent_constraint)
    
    
def edge_update_step(
    edge_feature: tf.Tensor, 
    edge_feature_prev: tf.Tensor,
    update_projection: Union[
        keras.layers.Dense, keras.layers.GRUCell, keras.layers.LSTMCell],
) -> tf.Tensor:
    if isinstance(update_projection, keras.layers.Dense):
        edge_feature_update = update_projection(
            tf.concat([edge_feature_prev, edge_feature], axis=1))
    else:
        edge_feature_update, _ = update_projection(
            inputs=edge_feature,
            states=edge_feature_prev)
    return edge_feature_update

def edge_message_step(
    edge_feature: tf.Tensor,
    edge_src: tf.Tensor,
    edge_dst: tf.Tensor,
) -> tf.Tensor:
    num_nodes = tf.maximum(tf.reduce_max(edge_src), tf.reduce_max(edge_dst)) + 1
    message = tf.math.unsorted_segment_sum(edge_feature, edge_dst, num_nodes)
    message = tf.gather(message, edge_src)
    message -= _get_reverse_edge_features(edge_feature, edge_src, edge_dst)
    return message

def _get_reverse_edge_features(edge_feature, edge_src, edge_dst):
    edge_exclude = tf.logical_and(
        edge_src[:, None] == edge_dst,
        edge_dst[:, None] == edge_src
    )
    edge_forward, edge_reverse = tf.split(tf.where(edge_exclude), 2, axis=-1)
    return tf.tensor_scatter_nd_add(
        tf.zeros_like(edge_feature), 
        tf.expand_dims(edge_forward, -1), 
        tf.gather(edge_feature, edge_reverse)
    )