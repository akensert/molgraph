import tensorflow as tf

import warnings 

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
    ...     discard_negative_values=False,
    ... )
    >>> maps = gam_model(esol['test']['x'].separate())

    References:
        .. [#] Pope et al. https://ieeexplore.ieee.org/document/8954227
    '''

    def __init__(
        self,
        model,
        layer_names: List[str] | None = None,
        output_activation: Optional[str] = None,
        discard_negative_values: bool = False,
        reduce_features: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(model, output_activation, **kwargs)
        self._discard_negative_values = discard_negative_values
        if layer_names is not None:
            warnings.warn(
                (
                    "`layer_names` is deprecated. All node features will be "
                    "considered by default."
                ),
                DeprecationWarning,
                stacklevel=2
            )
        self._layer_names = layer_names
        self._reduce_features = reduce_features

    def compute_saliency(
        self,
        x: GraphTensor,
        y: Optional[tf.Tensor]
    ) -> tf.Tensor:

        graph_indicator = None
        features = []
        with tf.GradientTape() as tape:

            for layer in self._model.layers:
                
                if not isinstance(x, GraphTensor):
                    x = layer(x)
                    continue 

                if graph_indicator is None:
                    graph_indicator = x.graph_indicator

                if hasattr(layer, '_taped_call'):
                    x, taped_features = layer._taped_call(x, tape)
                    features.extend(taped_features)
                else:
                    tape.watch(x.node_feature)
                    features.append(x.node_feature)
                    x = layer(x)
        
            predictions = self._activation(x)
            predictions = self._process_predictions(predictions, y)

        gradients = tape.gradient(predictions, features)
        features = tf.concat(features, axis=-1)
        gradients = tf.concat(gradients, axis=-1)

        alpha = tf.math.segment_mean(gradients, graph_indicator)
        alpha = tf.gather(alpha, graph_indicator)

        activation_maps = tf.where(gradients != 0, alpha * features, gradients)

        if self._reduce_features:
            activation_maps = tf.reduce_mean(activation_maps, axis=-1)
        
        if self._discard_negative_values:
            activation_maps = tf.nn.relu(activation_maps)

        return activation_maps

