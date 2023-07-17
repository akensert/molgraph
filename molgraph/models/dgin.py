import tensorflow as tf
from tensorflow import keras
from typing import Optional
from typing import Tuple
from typing import Union
from typing import Callable

from molgraph.internal import register_keras_serializable 

from molgraph.tensors.graph_tensor import GraphTensor
from molgraph.layers.message_passing.edge_conv import EdgeConv
from molgraph.layers.convolutional.gin_conv import GINConv
from molgraph.layers.readout.node_readout import NodeReadout


@register_keras_serializable(package='molgraph')
class DGIN(keras.layers.Layer):

    '''Directed graph isomorphism network (DGIN).

    Implementation based on Wieder et al. (2021) [#]_. 
    
    **Important:**

    As of now, EdgeConv only works on (sub)graphs with at least one edge/bond. 
    If your dataset consists of molecules with a single atom, please add self loops: 
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
    >>> x = molgraph.layers.Readout(name='readout')(x)
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
        edge_message_kwargs (dict, None):
            An optional dictionary of parameters which can be passed to the
            `EdgeConv` layers of this model. Note: as ``units`` and ``activation``
            are already specified, it will be dropped from the dict (if it exists 
            there). If None, an empty dict will be passed. Default to None.
        node_message_kwargs (dict, None):
            An optional dictionary of parameters which can be passed to the
            `GINConv` layers of this model. Note: as ``units`` and ``activation``
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
        edge_message_kwargs: Optional[dict] = None,
        node_message_kwargs: Optional[dict] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.units = units
        self.activation = keras.activations.get(activation)

        if edge_message_kwargs is None:
            edge_message_kwargs = {}
        else:
            edge_message_kwargs.pop('units', None)
            edge_message_kwargs.pop('activation', None)

        self.edge_message_kwargs = edge_message_kwargs

        if node_message_kwargs is None:
            node_message_kwargs = {}
        else:
            node_message_kwargs.pop('units', None)
            node_message_kwargs.pop('activation', None)
            
        self.node_message_kwargs = node_message_kwargs

        self.edge_message_steps = edge_message_steps
        self.node_message_steps = node_message_steps

    def build(self, input_shape: tf.TensorShape) -> None:

        if not self.units:
            self.units = input_shape[-1]

        self.edge_message_functions = [
            EdgeConv(
                units=self.units, 
                activation=self.activation, 
                **self.edge_message_kwargs
            ) for _ in range(self.edge_message_steps)
        ]
        self.node_message_functions = [
            GINConv(
                units=self.units, 
                activation=self.activation, 
                use_edge_features=False,
                **self.node_message_kwargs
            ) for _ in range(self.node_message_steps)
        ]
        self.node_readout = NodeReadout()

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
        
        x = tensor
        for edge_message_step in self.edge_message_functions:
            x = edge_message_step(x)

        # Pool messages (n_edges, dim) -> (n_nodes, dim)
        x = self.node_readout(x)
        
        # Concatenate initial node features and aggregated node features
        node_feature = tf.concat([
            tensor.node_feature, x.node_feature], axis=-1)
        
        x = x.update(node_feature=node_feature)
        for node_message_step in self.node_message_functions:
            x = node_message_step(x)

        return tensor_orig.update(
            node_feature=x.node_feature,
            edge_state=x.edge_state)


    def get_config(self) -> dict:
        base_config = super().get_config()
        config = {
            'units': self.units,
            'activation': keras.activations.serialize(self.activation),
            'edge_message_steps': self.edge_message_steps,
            'node_message_steps': self.node_message_steps,
            'edge_message_kwargs': self.edge_message_kwargs,
            'node_message_kwargs': self.node_message_kwargs,
        }
        base_config.update(config)
        return base_config
    
    