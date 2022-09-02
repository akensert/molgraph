import tensorflow as tf
from tensorflow import keras

from typing import List
from typing import Optional
from typing import Union

from molgraph.tensors.graph_tensor import GraphTensor


NOT_IMPLEMENTED_ERROR_MESSAGE = (
    "{} only makes predictions (call model.predict instead)")


@keras.utils.register_keras_serializable(package='molgraph')
class GradientActivationMapping(keras.Model):

    '''Gradient activation mapping.

    Implementation is based on Pope et al. [#]_.

    **Example:**

    >>> encoder = molgraph.chemistry.MolecularGraphEncoder(
    ...     atom_encoder=molgraph.chemistry.AtomicFeaturizer([
    ...         molgraph.chemistry.features.Symbol(),
    ...         molgraph.chemistry.features.Hybridization()
    ...     ])
    ... )
    >>> esol = molgraph.chemistry.datasets.get('esol')
    >>> esol['train']['x'] = encoder(esol['train']['x'])
    >>> esol['test']['x'] = encoder(esol['test']['x'])
    >>> # Pass GraphTensor to model
    >>> gnn_model = tf.keras.Sequential([
    ...     tf.keras.layers.Input(type_spec=esol['train']['x'].spec),
    ...     molgraph.layers.GCNConv(units=128, name='gcn_conv_1'),
    ...     molgraph.layers.GCNConv(units=128, name='gcn_conv_2'),
    ...     molgraph.layers.GCNConv(units=128, name='gcn_conv_3'),
    ...     molgraph.layers.Readout('mean'),
    ...     tf.keras.layers.Dense(units=512),
    ...     tf.keras.layers.Dense(units=1)
    ... ])
    >>> gnn_model.compile(optimizer='adam', loss='mse')
    >>> gnn_model.fit(esol['train']['x'], esol['train']['y'], epochs=10)
    >>> gam_model = molgraph.models.GradientActivationMapping(
    ...     model=gnn_model,
    ...     layer_names=['gcn_conv_1', 'gcn_conv_2', 'gcn_conv_3'],
    ...     discard_negative_values=False,
    ... )
    >>> # Interpretability models can only be predicted with
    >>> maps = gam_model.predict(esol['test']['x'])

    References:
        .. [#] Pope et al. https://ieeexplore.ieee.org/document/8954227

    '''

    def __init__(
        self,
        model,
        layer_names: List[str],
        output_activation: Optional[str] = None,
        discard_negative_values: bool = True,
        *args, **kwargs,
    ) -> None:
        super().__init__()
        self._model = model
        self._layer_names = layer_names
        self._activation = keras.activations.get(output_activation)
        self._discard_negative_values = discard_negative_values

    def compute_activation_maps(
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

            if y is not None:
                y = tf.cond(
                    tf.rank(y) < 2,
                    lambda: tf.expand_dims(y, -1),
                    lambda: y
                )
                predictions = tf.cond(
                    tf.shape(y)[-1] > 1,
                    lambda: tf.gather_nd(
                        predictions,
                        tf.stack([
                            tf.range(tf.shape(y)[0], dtype=tf.int64),
                            tf.argmax(y, axis=-1)
                        ], axis=1)
                    ),
                    lambda: predictions
                )

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

    def predict_step(self, data) -> tf.RaggedTensor:
        if isinstance(data, (GraphTensor)):
            x, y = data, None
        else:
            x, y = data[:2]

        activation_maps = self.compute_activation_maps(x, y)

        if isinstance(x.node_feature, tf.RaggedTensor):
            value_rowids = x.node_feature.value_rowids()
        else:
            value_rowids = x.graph_indicator

        nrows = tf.reduce_max(value_rowids) + 1

        activation_maps = tf.RaggedTensor.from_value_rowids(
            activation_maps, value_rowids, nrows=nrows)

        return activation_maps

    def train_step(self, data) -> None:
        raise NotImplementedError(
            NOT_IMPLEMENTED_ERROR_MESSAGE.format(self.__class__.__name__))

    def test_step(self, data) -> None:
        raise NotImplementedError(
            NOT_IMPLEMENTED_ERROR_MESSAGE.format(self.__class__.__name__))

    def compile(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            NOT_IMPLEMENTED_ERROR_MESSAGE.format(self.__class__.__name__))

    @tf.function
    def call(self, inputs, *args, **kwargs):
        return self.predict_step(inputs)

    def predict(
        self,
        x: Union[GraphTensor, tf.data.Dataset, tf.keras.utils.Sequence],
        batch_size: Optional[int] = None,
        verbose: int = 0,
        steps: Optional[int] = None,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
        max_queue_size: int = 10,
        workers: int = 1,
        use_multiprocessing: bool = False
    ) -> tf.RaggedTensor:
        '''Generates gradient activation maps'''
        return super().predict(
            x=x,
            batch_size=batch_size,
            verbose=verbose,
            steps=steps,
            callbacks=callbacks,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing
        )

    def get_config(self):
        # base_config = super().get_config()
        config = {
            'model': keras.layers.serialize(self._model),
            'layer_names': self._layer_names,
            'output_activation': keras.activations.serialize(self._activation),
            'discard_negative_values': self._discard_negative_values,
        }
        # base_config.update(config)
        # return base_config
        return config

    @classmethod
    def from_config(cls, config):
        config['model'] = keras.layers.deserialize(config['model'])
        return cls(**config)
