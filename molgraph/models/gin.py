import tensorflow as tf
from tensorflow import keras

from keras import regularizers
from keras import initializers 
from keras import constraints 

from typing import Optional
from typing import Union
from typing import Callable

from molgraph.internal import register_keras_serializable 

from molgraph.tensors.graph_tensor import GraphTensor
from molgraph.layers.convolutional.gin_conv import GINConv


@register_keras_serializable(package='molgraph')
class GIN(keras.layers.Layer):

    '''Graph isomorphism network (GIN), with or without edge features.

    Implementation based on Xu et al. (2019) [#]_ or Hu et al. (2020) [#]_.

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
    >>> # (n_nodes, n_features) -> (n_nodes, n_steps, n_features)
    >>> x = molgraph.models.GIN(steps=4, units=32, merge_mode='stack')(inputs)
    >>> # (n_nodes, n_steps, n_features) -> (n_graphs, n_steps, n_features)
    >>> outputs = molgraph.layers.Readout()(x)
    >>> gin_model = tf.keras.Model(inputs, outputs)
    >>> graph_embeddings = gin_model.predict(graph_tensor, verbose=0)
    >>> graph_embeddings.shape
    (2, 5, 32)
 
    Args:
        steps (int):
            Number of message passing steps. Default to 4.
        units (int, None):
            Number of hidden units of the message passing. Default to None.
        merge_mode (str):
            How the node embeddings should be merged. Either of 'concat', 
            'stack', 'sum', 'mean', 'max' or 'min'. Default to 'stack'.
        use_edge_features (bool):
            Whether or not to use edge features. Default to False.
        apply_relu_activation (bool):
            Whether to apply relu activation before aggregation step. Only relevant
            if use_edge_features is set to True. Default to False.
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
        .. [#] https://arxiv.org/pdf/1810.00826.pdf
        .. [#] https://arxiv.org/pdf/1905.12265.pdf
    '''
    
    def __init__(
        self,
        steps: int = 4,
        units: Optional[int] = None,
        merge_mode: str = 'stack',
        use_edge_features=True,
        apply_relu_activation=True,
        normalization: Union[None, str, bool] = None,
        dropout: Optional[float] = None,
        activation: Union[
            None, str, Callable[[tf.Tensor], tf.Tensor]] = None,
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
        self.merge_mode = merge_mode
        self.use_edge_features = use_edge_features
        self.apply_relu_activation = apply_relu_activation
        self.normalization = normalization
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

        self.initial_projection = self._get_dense()

        self.gin_convs = [
            self._get_gin_conv() for _ in range(self.steps)]

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
        
        tensor = tensor.update({
            'node_feature': self.initial_projection(tensor.node_feature)})
        
        x = tensor 

        node_features = [x.node_feature]
        for gin_conv in self.gin_convs:
            x = gin_conv(x)
            node_features.append(x.node_feature)

        merge_mode = self.merge_mode.lower()

        if merge_mode.startswith('stack'):
            node_feature = tf.stack(node_features, axis=-2) # axis -1 or -2?
        elif merge_mode.startswith('concat'):
            node_feature = tf.concat(node_features, axis=-1)
        else:
            reduce_fn = getattr(tf.math, f'reduce_{merge_mode}')
            node_feature = reduce_fn(node_features, axis=0)

        return tensor_orig.update(node_feature=node_feature)

    def get_config(self) -> dict:
        base_config = super().get_config()
        config = {
            'steps': self.steps,
            'units': self.units,
            'merge_mode': self.merge_mode,
            'use_edge_features': self.use_edge_features,
            'apply_relu_activation': self.apply_relu_activation,
            'normalization': self.normalization,
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
    
    def _get_gin_conv(self):
        gin_kwargs = dict(
            units=self.units,
            use_edge_features=self.use_edge_features,
            apply_relu_activation=self.apply_relu_activation,
            self_projection=False,
            residual=False,
            dropout=self.dropout,
            normalization=self.normalization,
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
        gin_kwargs["kernel_initializer"] = kernel_initializer
        gin_kwargs["bias_initializer"] = bias_initializer
        return GINConv(**gin_kwargs)
    
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