import tensorflow as tf
from tensorflow import keras

from typing import Optional
from typing import Tuple

from molgraph.internal import register_keras_serializable 

from molgraph.tensors.graph_tensor import GraphTensor
from molgraph.layers.message_passing.mpnn_conv import message_step


@register_keras_serializable(package='molgraph')
class MPNN(keras.layers.Layer):

    '''Message passing neural network (MPNN) with weight tying.

    Implementation is based on Gilmer et al. (2017) [#]_. 

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
    >>> x = molgraph.models.MPNN(units=32, steps=4, name='mpnn')(inputs)
    >>> x = molgraph.layers.SetGatherReadout(name='readout')(x)
    >>> outputs = tf.keras.layers.Dense(10, activation='sigmoid')(x)
    >>> mpnn_classifier = tf.keras.Model(inputs, outputs)
    >>> # Make predictions
    >>> preds = mpnn_classifier.predict(graph_tensor, verbose=0)
    >>> preds.shape
    (2, 10)

    Args:
        units (int, None):
            Number of hiden units of the message passing. These include the
            dense layers associated with the message functions, and GRU cells
            associated with the update functions. If None, hidden units are
            set to the input dimension. Default to None.
        steps (int):
            Number of message passing steps. Default to 4.
        residual: (bool)
            Whether to add skip connection to the output of each step.
            Default to True.
        dropout: (float, None):
            Dropout applied to the output of step. Default to None.
        message_kwargs (dict, None):
            An optional dictionary of parameters which can be passed to the
            dense layers of the message functions. Note: as ``units`` is already
            specified, it will be dropped from the dict (if it exists there).
            If None, an empty dict will be passed. Default to None.
        update_kwargs (dict, None):
            An optional dictionary of parameters which can be passed to the
            GRUCells of the update functions. Note: as ``units`` is already
            specified, it will be dropped from the dict (if it exists there).
            If None, an empty dict will be passed. Default to None.

    References:
        .. [#] https://arxiv.org/pdf/1704.01212.pdf
    '''

    def __init__(
        self,
        steps: int = 4,
        units: Optional[int] = None,
        residual: bool = True,
        dropout: Optional[float] = None,
        message_kwargs: Optional[dict] = None,
        update_kwargs: Optional[dict] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.steps = steps
        self.residual = residual
        self.dropout = (
            None if dropout is None else keras.layers.Dropout(dropout))
        self.message_kwargs = {} if message_kwargs is None else message_kwargs
        self.update_kwargs = {} if update_kwargs is None else update_kwargs
        self.message_kwargs.pop('units', None)
        self.update_kwargs.pop('units', None)

    def build(self, input_shape: Tuple[int, ...]) -> None:
        if self.units is None:
            self.units = input_shape[-1]

        if self.units != input_shape[-1]:
            self.node_resample = keras.layers.Dense(self.units, use_bias=False)

        self.message_projection = keras.layers.Dense(
            units=self.units*self.units, **self.message_kwargs)
        self.update_step = keras.layers.GRUCell(
            units=self.units, **self.update_kwargs)
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

        edge_src = tensor.edge_src
        edge_dst = tensor.edge_dst
        node_feature_updated = tensor.node_feature
        # MPNN requires edge features, if edge features do not exist,
        # we initialize a ones vector.
        if tensor.edge_feature is None:
            edge_feature = tf.ones(
                shape=[tf.shape(edge_src)[0], 1],
                dtype=node_feature_updated.dtype)
        else:
            edge_feature = tensor.edge_feature

        if hasattr(self, 'node_resample'):
            node_feature_updated = self.node_resample(node_feature_updated)

        for _ in range(self.steps):
            if self.residual:
                node_feature_residual = node_feature_updated
            # Perform a step of message passing (message function)
            node_feature_aggregated = message_step(
                node_feature=node_feature_updated,
                edge_feature=edge_feature,
                edge_src=edge_src,
                edge_dst=edge_dst,
                projection=self.message_projection)
            # Perform a step of GRU (update function)
            node_feature_updated, _ = self.update_step(
                inputs=node_feature_aggregated,
                states=node_feature_updated)

            if self.residual:
                node_feature_updated += node_feature_residual

            if self.dropout is not None:
                node_feature_updated = self.dropout(node_feature_updated)

        return tensor_orig.update({'node_feature': node_feature_updated})

    def get_config(self) -> dict:
        base_config = super().get_config()
        config = {
            'units': self.units,
            'steps': self.steps,
            'residual': self.residual,
            'dropout': self.dropout,
            'message_kwargs': self.message_kwargs,
            'update_kwargs': self.update_kwargs,
        }
        base_config.update(config)
        return base_config
