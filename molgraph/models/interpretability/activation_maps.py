import tensorflow as tf

from typing import List
from typing import Optional

from molgraph.layers.gnn_layer import GNNLayer
from molgraph.tensors.graph_tensor import GraphTensor
from molgraph.models.interpretability.saliency import SaliencyMapping


class GradientActivationMapping(SaliencyMapping):

    '''Gradient activation mapping.

    Implementation is based on Pope et al. [#]_.

    Alias: ``GradientActivation``

    Example usage:

    >>> encoder = molgraph.chemistry.MolecularGraphEncoder(
    ...     atom_encoder=molgraph.chemistry.Featurizer([
    ...         molgraph.chemistry.features.Symbol(),
    ...         molgraph.chemistry.features.Hybridization()
    ...     ])
    ... )
    >>> esol = molgraph.chemistry.datasets.get('esol')
    >>> esol['train']['x'] = encoder(esol['train']['x'])
    >>> esol['test']['x'] = encoder(esol['test']['x'])
    ...
    >>> gnn_model = tf.keras.Sequential([
    ...     molgraph.layers.GCNConv(units=128, name='gcn_conv_1'),
    ...     molgraph.layers.GCNConv(units=128, name='gcn_conv_2'),
    ...     molgraph.layers.GCNConv(units=128, name='gcn_conv_3'),
    ...     molgraph.layers.Readout('mean'),
    ...     tf.keras.layers.Dense(units=512),
    ...     tf.keras.layers.Dense(units=1)
    ... ])
    ...
    >>> gnn_model.compile(optimizer='adam', loss='mse')
    >>> _ = gnn_model.fit(
    ...     esol['train']['x'], esol['train']['y'], epochs=10, verbose=0)
    ...
    >>> gam_model = molgraph.models.GradientActivationMapping(
    ...     model=gnn_model,
    ...     layer_names=['gcn_conv_1', 'gcn_conv_2', 'gcn_conv_3'],
    ...     discard_negative_values=False,
    ... )
    >>> maps = gam_model(esol['test']['x'].separate())

    References:
        .. [#] Pope et al. https://ieeexplore.ieee.org/document/8954227
    '''

    def __init__(
        self,
        model,
        layer_names: List[str] = None,
        output_activation: Optional[str] = None,
        discard_negative_values: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(model, output_activation, **kwargs)
        if layer_names is not None:
            self._layer_names = layer_names
        else:
            layer_names = []
            for layer in model.layers:
                if isinstance(layer, GNNLayer):
                    layer_names.append(layer.name)
            if not layer_names:
                raise ValueError(
                    'Could not obtain GNN layer(s).')
            self._layer_names = layer_names
        self._discard_negative_values = discard_negative_values

    def compute_saliency(
        self,
        x: GraphTensor,
        y: Optional[tf.Tensor]
    ) -> tf.Tensor:

        features = []
        with tf.GradientTape() as tape:

            for layer in self._model.layers:
                x = layer(x)
                if layer.name in self._layer_names:
                    node_feature = x.node_feature
                    if isinstance(x.node_feature, tf.RaggedTensor):
                        node_feature = node_feature.flat_values
                    features.append(node_feature)

            predictions = self._activation(x)
            predictions = self._process_predictions(predictions, y)

        gradients = tape.gradient(predictions, features)
        features = tf.stack(features, axis=0)
        gradients = tf.stack(gradients, axis=0)

        alpha = tf.reduce_mean(gradients, axis=1, keepdims=True)

        activation_maps = tf.where(gradients != 0, alpha * features, gradients)
        activation_maps = tf.reduce_mean(activation_maps, axis=-1)

        if self._discard_negative_values:
            activation_maps = tf.nn.relu(activation_maps)

        activation_maps = tf.reduce_mean(activation_maps, axis=0)

        return activation_maps

