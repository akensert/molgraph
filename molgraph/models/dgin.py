import tensorflow as tf
from tensorflow import keras
from typing import Optional
from typing import Tuple
from typing import Union
from typing import Callable

from molgraph.tensors.graph_tensor import GraphTensor
from molgraph.layers.gnn_ops import propagate_node_features
from molgraph.layers.message_passing.edge_conv import edge_message_step


@keras.utils.register_keras_serializable(package='molgraph')
class DGIN(keras.models.Model):

    '''Directed graph isomorphism network (DGIN).

    Implementation is based on Wieder et al. (2021) [#]_. 
    
    **Important:**

    As of now, EdgeConv only works on (sub)graphs with at least one edge/bond. If your dataset consists
    of molecules with a single atom, please add self loops: 
    ``molgraph.chemistry.MolecularGraphEncoder(..., self_loops=True)``
    
    **Example:**

    >>> # Obtain GraphTensor
    >>> graph_tensor = molgraph.GraphTensor(
    ...     data={
    ...         'edge_src': [[1, 0], [1, 2, 0, 2, 1, 0]],
    ...         'edge_dst': [[0, 1], [0, 0, 1, 1, 2, 2]],
    ...         'node_feature': [
    ...             [[1.0, 0.0], [1.0, 0.0]],
    ...             [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    ...         ],
    ...     }
    ... )
    >>> # Build Functional model
    >>> inputs = tf.keras.layers.Input(type_spec=graph_tensor.unspecific_spec)
    >>> x = molgraph.models.DGIN(units=32, name='dgin')(inputs)
    >>> x = molgraph.layers.SetGatherReadout(name='readout')(x)
    >>> outputs = tf.keras.layers.Dense(10, activation='sigmoid')(x)
    >>> dgin_classifier = tf.keras.Model(inputs, outputs)
    >>> # Make predictions
    >>> preds = dgin_classifier.predict(graph_tensor, verbose=0)
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
        edge_message_steps (int):
            Number of edge message passing steps. Default to 4.
        node_message_steps (int):
            Number of node message passing steps. Default to 4.
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
        .. [#] https://www.mdpi.com/1420-3049/26/20/6185
    '''
    
    def __init__(
        self,
        units: Optional[int] = None,
        activation: Union[
            None, str, Callable[[tf.Tensor], tf.Tensor]] = 'relu',
        edge_message_steps: int = 4,
        node_message_steps: int = 4,
        dropout: Optional[float] = None,
        parallel_iterations: Optional[int] = None,
        dense_kwargs: Optional[dict] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        self.edge_message_steps = edge_message_steps
        self.node_message_steps = node_message_steps
        self.dropout = (
            None if dropout is None else keras.layers.Dropout(dropout))
        self.parallel_iterations = parallel_iterations
        self.dense_kwargs = {} if dense_kwargs is None else dense_kwargs
        self.dense_kwargs.pop('units', None)
        self.dense_kwargs.pop('activaton', None)

    def build(self, input_shape: Optional[Tuple[int, ...]] = None) -> None:
        if not self.units:
            self.units = input_shape[-1]
        self.initial_projection = keras.layers.Dense(
            self.units, self.activation, **self.dense_kwargs)
        self.update_projection = keras.layers.Dense(
            self.units, **self.dense_kwargs)
        self.node_projection = keras.layers.Dense(
            self.units + input_shape[-1], **self.dense_kwargs)
        self.epsilon = tf.Variable(0.)
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
        if tensor.edge_feature is None:
            tensor = tensor.update({
                'edge_feature': tf.ones(
                    shape=[tf.shape(tensor.edge_dst)[0], 1],
                    dtype=tensor.node_feature.dtype)})
            
        edge_feature = tf.gather(tensor.node_feature, tensor.edge_src)
        edge_feature = tf.concat([edge_feature, tensor.edge_feature], axis=-1)
        edge_feature = self.initial_projection(edge_feature)
        
        tensor = tensor.update({'edge_feature': edge_feature})
        
        for _ in range(self.edge_message_steps):
            
            message = edge_message_step(
                edge_feature=tensor.edge_feature,
                edge_src=tensor.edge_src,
                edge_dst=tensor.edge_dst)

            edge_feature = self.activation(
                self.update_projection(message) + edge_feature)
            
            if self.dropout is not None:
                edge_feature = self.dropout(edge_feature)

            # TODO: should skip connection be applied? 
            tensor = tensor.update({'edge_feature': edge_feature})

        message = tf.math.unsorted_segment_sum(
            tensor.edge_feature, tensor.edge_dst, tf.shape(tensor.node_feature)[0]) 
        
        node_feature = tf.concat([tensor.node_feature, message], axis=-1)
        
        node_feature_initial = self.node_projection(
            node_feature * (1 + self.epsilon))
        
        for _ in range(self.node_message_steps):
            node_feature = propagate_node_features(
                node_feature=node_feature, 
                edge_src=tensor.edge_src,
                edge_dst=tensor.edge_dst)
            node_feature = node_feature + node_feature_initial
            
        return tensor_orig.update({'node_feature': node_feature})

    def get_config(self) -> dict:
        base_config = super().get_config()
        config = {
            'units': self.units,
            'activation': keras.activations.serialize(self.activation),
            'edge_message_steps': self.edge_message_steps,
            'node_message_steps': self.node_message_steps,
            'dropout': self.dropout,
            'parallel_iterations': self.parallel_iterations,
            'dense_kwargs': self.dense_kwargs,
        }
        base_config.update(config)
        return base_config
    
    