import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import regularizers
from tensorflow.keras import initializers 
from tensorflow.keras import constraints 

from typing import Optional
from typing import Union
from typing import Callable

from molgraph.internal import register_keras_serializable 

from molgraph.tensors.graph_tensor import GraphTensor
from molgraph.layers.message_passing.edge_conv import EdgeConv
from molgraph.layers.readout.node_readout import NodeReadout


@register_keras_serializable(package='molgraph')
class DMPNN(keras.layers.Layer):

    '''Directed message passing neural network (DMPNN).

    Implementation based on Yang et al. (2019) [#]_.

    As of now, DMPNN only works on (sub)graphs with at least one edge/bond. 
    If your dataset consists of molecules with single atoms, please add self 
    loops: ``molgraph.chemistry.MolecularGraphEncoder(..., self_loops=True)``. 

    Example usage:

    >>> # Obtain GraphTensor
    >>> graph_tensor = molgraph.GraphTensor(
    ...     sizes=[2, 3],
    ...     node_feature=[[1., 0.], [1., 0.], [1., 0.], [1., 0.], [0., 1.]],
    ...     edge_feature=[[1., 0.], [0., 1.], [0., 1.], [0., 1.], 
    ...                   [1., 0.], [0., 1.], [1., 0.], [0., 1.]],
    ...     edge_src=[1, 0, 3, 4, 2, 4, 3, 2],
    ...     edge_dst=[0, 1, 2, 2, 3, 3, 4, 4],
    ... )
    >>> # Build Functional model
    >>> inputs = tf.keras.layers.Input(type_spec=graph_tensor.spec)
    >>> x = molgraph.models.DMPNN(units=32, name='dmpnn')(inputs)
    >>> x = molgraph.layers.Readout(name='readout')(x)
    >>> outputs = tf.keras.layers.Dense(10, activation='sigmoid')(x)
    >>> dmpnn_classifier = tf.keras.Model(inputs, outputs)
    >>> # Make predictions
    >>> preds = dmpnn_classifier.predict(graph_tensor, verbose=0)
    >>> preds.shape
    (2, 10)
 
    Args:
        steps (int):
            Number of message passing steps. Default to 4.
        units (int, None):
            Number of hidden units of the message passing. Default to None.
        normalization: (None, str, bool):
            Whether to apply layer normalization to the output. If batch 
            normalization is desired, pass 'batch_norm'. Default to True.
        residual: (bool)
            Whether to add skip connection to the output. Default to True.
        dropout: (float, None):
            Dropout applied to the output of the layer. Default to None.
        activation (tf.keras.activations.Activation, callable, str, None):
            Activation function applied to the projections. Default to 'relu'.
        use_bias (bool):
            Whether the layer should use biases. Default to True.
        kernel_initializer (tf.keras.initializers.Initializer, str):
            Initializer function for the kernels. If None,
            ``tf.keras.initializers.TruncatedNormal(stddev=0.005)`` will be 
            used. Default to None.
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
        .. [#] https://arxiv.org/abs/1904.01561
    '''
    
    def __init__(
        self,
        steps: int = 4,
        units: Optional[int] = None,
        normalization: Union[None, str, bool] = 'layer_norm',
        residual: bool = True,
        dropout: Optional[float] = None,
        activation: Union[
            None, str, Callable[[tf.Tensor], tf.Tensor]] = 'relu',
        use_bias: Optional[bool] = True,
        kernel_initializer: Union[str, None, initializers.Initializer] = None,
        bias_initializer: Union[str, None, initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_regularizer: Optional[regularizers.Regularizer] = None,
        activity_regularizer: Optional[regularizers.Regularizer] = None,
        kernel_constraint: Optional[constraints.Constraint] = None,
        bias_constraint: Optional[constraints.Constraint] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.steps = steps
        self.units = units
        self.normalization = normalization
        self.residual = residual
        self.dropout = dropout
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        if kernel_initializer is None:
            kernel_initializer = initializers.TruncatedNormal(stddev=0.005)
        self.kernel_initializer  = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer or 'zeros')
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)


    def build(self, input_shape: tf.TensorShape) -> None:
        
        if not self.units:
            self.units = input_shape[-1]

        self.edge_convs = [
            self._get_edge_conv() for _ in range(self.steps)]

        self.node_readout = NodeReadout()

        self.projection = self._get_dense()

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
        for edge_conv in self.edge_convs:
            x = edge_conv(x)

        # Pool messages (n_edges, dim) -> (n_nodes, dim)
        x = self.node_readout(x)
        
        # Concatenate initial node features and aggregated node features
        node_feature = tf.concat([
            tensor.node_feature, x.node_feature], axis=-1)
        
        node_feature = self.projection(node_feature)

        return tensor_orig.update(
            node_feature=node_feature,
            edge_state=x.edge_state)


    def get_config(self) -> dict:
        base_config = super().get_config()
        config = {
            'steps': self.steps,
            'units': self.units,
            'normalization': self.normalization,
            'residual': self.residual,
            'dropout': self.dropout,
            'activation': keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
        }
        base_config.update(config)
        return base_config
    
    def _get_edge_conv(self):
        common_kwargs = dict(
            units=self.units,
            activation=self.activation,
            update_mode='dense',
            update_fn=None,
            use_bias=self.use_bias,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint)
        kernel_initializer = self.kernel_initializer.__class__.from_config(
            self.kernel_initializer.get_config())
        bias_initializer = self.bias_initializer.__class__.from_config(
            self.bias_initializer.get_config())
        common_kwargs["kernel_initializer"] = kernel_initializer
        common_kwargs["bias_initializer"] = bias_initializer
        return EdgeConv(**common_kwargs)
    
    def _get_dense(self):
        common_kwargs = dict(
            units=self.units,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint)
        kernel_initializer = self.kernel_initializer.__class__.from_config(
            self.kernel_initializer.get_config())
        bias_initializer = self.bias_initializer.__class__.from_config(
            self.bias_initializer.get_config())
        common_kwargs["kernel_initializer"] = kernel_initializer
        common_kwargs["bias_initializer"] = bias_initializer
        return keras.layers.Dense(**common_kwargs)