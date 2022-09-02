import tensorflow as tf
from tensorflow import keras

from typing import Optional
from typing import Union
from typing import Callable
from typing import List

from molgraph.tensors.graph_tensor import GraphTensor


NOT_IMPLEMENTED_ERROR_MESSAGE = (
    "{} only makes predictions (call model.predict instead)")


@keras.utils.register_keras_serializable(package='molgraph')
class SaliencyMapping(keras.Model):
    '''Vanilla saliency mapping.

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
    >>> saliency = molgraph.models.SaliencyMapping(model=gnn_model)
    >>> # Interpretability models can only be predicted with
    >>> saliency_maps = saliency.predict(esol['test']['x'])

    '''

    def __init__(
        self,
        model: keras.Model,
        output_activation: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._model = model
        self._activation = keras.activations.get(output_activation)

    def compute_gradients(self, x: GraphTensor, y: tf.Tensor) -> tf.Tensor:

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x.node_feature)
            predictions = self._model(x)
            predictions = self._activation(predictions)

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

        return tape.gradient(predictions, _maybe_flat_values(x.node_feature))

    def predict_step(self, data) -> tf.RaggedTensor:

        if isinstance(data, (GraphTensor)):
            x, y = data, None
        else:
            x, y = data[:2]

        saliency = self.compute_saliency(x, y)

        if isinstance(x.node_feature, tf.RaggedTensor):
            value_rowids = x.node_feature.value_rowids()
        else:
            value_rowids = x.graph_indicator

        nrows = tf.reduce_max(value_rowids) + 1

        saliency = tf.RaggedTensor.from_value_rowids(
            saliency, value_rowids, nrows=nrows)

        return saliency

    def train_step(self, data) -> None:
        raise NotImplementedError(
            NOT_IMPLEMENTED_ERROR_MESSAGE.format(self.__class__.__name__))

    def test_step(self, data) -> None:
        raise NotImplementedError(
            NOT_IMPLEMENTED_ERROR_MESSAGE.format(self.__class__.__name__))

    def compile(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            NOT_IMPLEMENTED_ERROR_MESSAGE.format(self.__class__.__name__))

    def compute_saliency(self, x: GraphTensor, y: tf.Tensor) -> tf.Tensor:
        gradients = self.compute_gradients(x, y)
        gradients = tf.abs(gradients)
        return tf.reduce_sum(gradients, axis=1)

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
        '''Generates saliency maps'''
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
        config = {
            'model': keras.layers.serialize(self._model),
            'output_activation': keras.activations.serialize(self._activation),
        }
        return config

    @classmethod
    def from_config(cls, config):
        config['model'] = keras.layers.deserialize(config['model'])
        return cls(**config)


@keras.utils.register_keras_serializable(package='molgraph')
class IntegratedSaliencyMapping(SaliencyMapping):
    '''Integrated saliency mapping.

    **Example:**

    >>> encoder = molgraph.chemistry.MolecularGraphEncoder(
    ...     atom_encoder=molgraph.chemistry.AtomicFeaturizer([
    ...         molgraph.chemistry.features.Symbol(),
    ...         molgraph.chemistry.features.Hybridization()
    ...     ])
    ... )
    >>> bbbp = molgraph.chemistry.datasets.get('bbbp')
    >>> bbbp['train']['x'] = encoder(bbbp['train']['x'])
    >>> bbbp['test']['x'] = encoder(bbbp['test']['x'])
    >>> # Pass GraphTensor to model
    >>> gnn_model = tf.keras.Sequential([
    ...     tf.keras.layers.Input(type_spec=bbbp['train']['x'].spec),
    ...     molgraph.layers.GCNConv(units=128, name='gcn_conv_1'),
    ...     molgraph.layers.GCNConv(units=128, name='gcn_conv_2'),
    ...     molgraph.layers.GCNConv(units=128, name='gcn_conv_3'),
    ...     molgraph.layers.Readout('mean'),
    ...     tf.keras.layers.Dense(units=512),
    ...     tf.keras.layers.Dense(units=1, activation='sigmoid')
    ... ])
    >>> gnn_model.compile(optimizer='adam', loss='mse')
    >>> gnn_model.fit(bbbp['train']['x'], bbbp['train']['y'], epochs=10)
    >>> saliency = molgraph.models.IntegratedSaliencyMapping(model=gnn_model)
    >>> # Interpretability models can only be predicted with
    >>> saliency_maps = saliency.predict(bbbp['test']['x'])

    '''

    def __init__(
        self,
        model: keras.Model,
        output_activation: Union[
            None, str, Callable[[tf.Tensor], tf.Tensor]] = None,
        steps: int = 20
    ) -> None:
        super().__init__(
            model=model,
            output_activation=output_activation)
        self.steps = steps

    def get_config(self):
        base_config = super().get_config()
        config = {
            'steps': self.steps,
        }
        base_config.update(config)
        return base_config

    @tf.function
    def compute_saliency(self, x: GraphTensor, y: tf.Tensor) -> tf.Tensor:

        original = _maybe_flat_values(x.node_feature)
        baseline = tf.zeros_like(original)
        alpha = tf.linspace(start=0.1, stop=1.0, num=self.steps+1)
        alpha = tf.cast(alpha, dtype=original.dtype)

        gradients_batch = tf.TensorArray(original.dtype, size=self.steps+1)

        def body(x, gradients_batch, i):
            step = baseline + alpha[i] * (original - baseline)
            x = x.update({'node_feature': step})
            gradients = self.compute_gradients(x, y)
            gradients_batch = gradients_batch.write(i, gradients)
            return x, gradients_batch, tf.add(i, 1)

        i = tf.constant(0)
        condition = lambda x, gradients_batch, i: tf.less(i, self.steps + 1)
        x, gradients_batch, i = tf.while_loop(
            cond=condition, body=body, loop_vars=[x, gradients_batch, i])

        gradients_batch = gradients_batch.stack()

        integrated_gradients = (gradients_batch[:-1] + gradients_batch[1:]) / 2.0
        integrated_gradients = tf.math.reduce_mean(integrated_gradients, axis=0)

        integrated_gradients = (original - baseline) * integrated_gradients

        return tf.reduce_sum(tf.abs(integrated_gradients), axis=1)


@keras.utils.register_keras_serializable(package='molgraph')
class SmoothGradSaliencyMapping(SaliencyMapping):
    '''Smooth-gradient saliency mapping.

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
    >>> saliency = molgraph.models.SmoothGradSaliencyMapping(model=gnn_model)
    >>> # Interpretability models can only be predicted with
    >>> saliency_maps = saliency.predict(esol['test']['x'])

    '''
    def __init__(
        self,
        model: keras.Model,
        output_activation: Union[
            None, str, Callable[[tf.Tensor], tf.Tensor]] = None,
        steps: int = 50,
        noise: float = 0.1,
    ) -> None:
        super().__init__(
            model=model,
            output_activation=output_activation)
        self.steps = steps
        self.noise = noise

    def get_config(self):
        base_config = super().get_config()
        config = {
            'steps': self.steps,
            'noise': self.noise,
        }
        base_config.update(config)
        return base_config

    def compute_saliency(self, x: GraphTensor, y: tf.Tensor) -> tf.Tensor:

        original = _maybe_flat_values(x.node_feature)

        gradients_batch = tf.TensorArray(original.dtype, size=self.steps)

        def body(x, gradients_batch, i):
            noisy = original + tf.random.normal(
                shape=(1, tf.shape(original)[1]),
                mean=0.0,
                stddev=self.noise,
                dtype=original.dtype)
            x = x.update({'node_feature': noisy})
            gradients = self.compute_gradients(x, y)
            gradients_batch = gradients_batch.write(i, tf.abs(gradients))
            return x, gradients_batch, tf.add(i, 1)

        i = tf.constant(0)
        condition = lambda x, gradients_batch, i: tf.less(i, self.steps)
        x, gradients_batch, i = tf.while_loop(
            cond=condition, body=body, loop_vars=[x, gradients_batch, i])

        gradients_batch = gradients_batch.stack()

        gradients_average = tf.math.reduce_mean(gradients_batch, axis=0)

        return tf.reduce_sum(gradients_average, axis=1)


def _maybe_flat_values(x):
    if isinstance(x, tf.RaggedTensor):
        return x.flat_values
    return x
