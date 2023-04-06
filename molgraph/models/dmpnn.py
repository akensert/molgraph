import tensorflow as tf
from tensorflow import keras
from typing import Optional
from typing import Tuple
from typing import Union
from typing import Callable

from molgraph.tensors import GraphTensor


@keras.utils.register_keras_serializable(package='molgraph')
class DMPNN(keras.layers.Layer):

    '''Directed message passing neural network (DMPNN).

    Implementation is based on Yang et al. (2019) [#]_.

    **Example:**

    >>> # Obtain GraphTensor
    >>> graph_tensor = molgraph.GraphTensor(
    ...     data={
    ...         'edge_dst': [[0, 1], [0, 0, 1, 1, 2, 2]],
    ...         'edge_src': [[1, 0], [1, 2, 0, 2, 1, 0]],
    ...         'node_feature': [
    ...             [[1.0, 0.0], [1.0, 0.0]],
    ...             [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    ...         ],
    ...     }
    ... )
    >>> # Build Functional model
    >>> inputs = tf.keras.layers.Input(type_spec=graph_tensor.unspecific_spec)
    >>> x = molgraph.models.MPNN(units=32, steps=4, name='dmpnn')(inputs)
    >>> x = molgraph.layers.SetGatherReadout(name='readout')(x)
    >>> outputs = tf.keras.layers.Dense(10, activation='sigmoid')(x)
    >>> dmpnn_classifier = tf.keras.Model(inputs, outputs)
    >>> # Make predictions
    >>> preds = dmpnn_classifier.predict(graph_tensor)
    >>> preds.shape
    (2, 10)
 
    Args:
        units (int, None):
            Number of hiden units of the message passing. These include the
            dense layers associated with the message functions, and GRU cells
            associated with the update functions. If None, hidden units are
            set to the input dimension. Default to None.
        activation (tf.keras.activations.Activation, callable, str, None):
            Activation function applied to the projections. Default to 'relu'.
        steps (int):
            Number of message passing steps. Default to 4.
        dropout: (float, None):
            Dropout applied to the output of step. Default to None.
        parallel_iterations (int, None):
            Number of ``parallel_iterations`` to be set for ``tf.map_fn`` to find
            the reverse edge features to be subtracted from the aggregated edge features.
        dense_kwargs (dict, None):
            An optional dictionary of parameters which can be passed to the
            dense layers of this model. Note: as ``units`` and ``activation``
            are already specified, it will be dropped from the dict (if it exists 
            there). If None, an empty dict will be passed. Default to None.

    References:
        .. [#] https://arxiv.org/pdf/1904.01561.pdf
    '''
    
    def __init__(
        self,
        units: Optional[int] = None,
        activation: Union[
            None, str, Callable[[tf.Tensor], tf.Tensor]] = 'relu',
        steps: int = 4,
        dropout: Optional[float] = None,
        parallel_iterations: Optional[int] = None,
        dense_kwargs: Optional[dict] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        self.steps = steps
        self.dropout = (
            None if dropout is None else keras.layers.Dropout(dropout))
        self.parallel_iterations = parallel_iterations
        self.dense_kwargs = {} if dense_kwargs is None else dense_kwargs
        self.dense_kwargs.pop('units', None)
        self.dense_kwargs.pop('activaton', None)

    def build(self, input_shape: Optional[Tuple[int, ...]] = None) -> None:
        self.initial_projection = keras.layers.Dense(
            self.units, self.activation, **self.dense_kwargs)
        self.update_projection = keras.layers.Dense(
            self.units, **self.dense_kwargs)
        self.final_projection = keras.layers.Dense(
            self.units, self.activation, **self.dense_kwargs)
        self.built = True
    
    def call(self, tensor: GraphTensor) -> GraphTensor:

        '''Defines the computation from inputs to outputs.
        
        This method should not be called directly, but indirectly
        via ``__call__()``. Upon first call, the layer is automatically
        built via ``build()``.
        
        Args:
            tensor (GraphTensor):
                Input to the layer.
                
        Returns:
            GraphTensor:
                A ``GraphTensor`` with updated node features.
        '''
        
        tensor_orig = tensor
        if isinstance(tensor.node_feature, tf.RaggedTensor):
            tensor = tensor.merge()
        
        # MPNN requires edge features, if edge features do not exist,
        # we initialize a ones vector.
        if not hasattr(tensor, 'edge_feature'):
            tensor = tensor.update({
                'edge_feature': tf.ones(
                    shape=[tf.shape(tensor.edge_dst)[0], 1],
                    dtype=tensor.node_feature.dtype)})
            
        edge_feature = tf.gather(tensor.node_feature, tensor.edge_src)
        edge_feature = tf.concat([edge_feature, tensor.edge_feature], axis=-1)
        edge_feature = self.initial_projection(edge_feature)
        
        tensor = tensor.update({'edge_feature': edge_feature})
        
        for _ in range(self.steps):
            
            message = edge_message_passing(
                tensor, self.parallel_iterations)

            edge_feature = self.activation(
                self.update_projection(message) + edge_feature)
            
            if self.dropout is not None:
                edge_feature = self.dropout(edge_feature)

            # TODO: should skip connection be applied? 
            tensor = tensor.update({'edge_feature': edge_feature})

        message = tf.math.unsorted_segment_sum(
            tensor.edge_feature, tensor.edge_dst, tf.shape(tensor.node_feature)[0]) 
            
        node_feature = self.final_projection(
            tf.concat([tensor.node_feature, message], axis=-1))
        
        return tensor_orig.update({'node_feature': node_feature})

    def get_config(self) -> dict:
        base_config = super().get_config()
        config = {
            'units': self.units,
            'activation': keras.activations.serialize(self.activation),
            'steps': self.steps,
            'dropout': self.dropout,
            'parallel_iterations': self.parallel_iterations,
            'dense_kwargs': self.dense_kwargs,
        }
        base_config.update(config)
        return base_config
    
    
def _get_reverse_edge_features_fn(
    tensor: GraphTensor
) -> tf.Tensor:
    # Find the index of "reverse" edge of edge_src->edge_dst
    edge_exclude = tf.logical_and(
        tensor.edge_src[:, None] == tensor.edge_dst,
        tensor.edge_dst[:, None] == tensor.edge_src)
    # Obtain index of edge_src->edge_dst ("forward") and its corresponding
    # edge_src<-edge_dst ("reverse"). For molecules: forward and reverse
    # edges are usually the same and always exist. 
    edge_forward, edge_reverse = tf.split(tf.where(edge_exclude), 2, axis=-1)
    return tf.tensor_scatter_nd_add(
        tf.zeros_like(tensor.edge_feature), 
        tf.expand_dims(edge_forward, -1), 
        tf.gather(tensor.edge_feature, edge_reverse))

def _get_reverse_edge_features(
    tensor: GraphTensor, 
    parallel_iterations: Optional[int] = None
) -> tf.Tensor:
    'Obtain reverse edge features to subtract from aggregated edge features.'
    output_spec = tf.RaggedTensorSpec(
        shape=[None, tensor.edge_feature.shape[-1]], 
        ragged_rank=0, 
        dtype=tf.float32)
    reverse_edge_feature = tf.map_fn(
        fn=_get_reverse_edge_features_fn, 
        elems=tensor.separate(), 
        fn_output_signature=output_spec,
        parallel_iterations=parallel_iterations)
    return reverse_edge_feature.merge_dims(outer_axis=0, inner_axis=1)
    
def edge_message_passing(
    tensor: GraphTensor, 
    parallel_iterations: Optional[int] = None
) -> tf.Tensor:
    num_nodes = tf.shape(tensor.node_feature)[0]
    message = tf.math.unsorted_segment_sum(
        tensor.edge_feature, tensor.edge_dst, num_nodes)
    message = tf.gather(message, tensor.edge_src)
    message = message - _get_reverse_edge_features(
        tensor, parallel_iterations)
    return message
    