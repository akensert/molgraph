import tensorflow as tf
from tensorflow import keras

from typing import Optional
from typing import Union
from typing import Callable

from molgraph.tensors.graph_tensor import GraphTensor


def automatically_infer_input_signature(func):
    def call(
        self: 'SaliencyMapping', 
        x: GraphTensor, 
        y: Optional[tf.Tensor] = None
    ) -> Union[tf.Tensor, tf.RaggedTensor]:
        if not self._built:
            self.build(x, y)
        return func(self, x, y)
    return call


class SaliencyMapping(tf.Module):
    '''Vanilla saliency mapping.

    Alias: ``Saliency``
    
    Example usage:

    >>> encoder = molgraph.chemistry.MolecularGraphEncoder(
    ...     atom_encoder=molgraph.chemistry.Featurizer([
    ...         molgraph.chemistry.features.Symbol(),
    ...         molgraph.chemistry.features.Hybridization()
    ...     ])
    ... )
    ...
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
    >>> saliency = molgraph.models.SaliencyMapping(model=gnn_model)
    >>> saliency_maps = saliency(esol['test']['x'].separate())
    '''

    def __init__(
        self,
        model: keras.Model,
        output_activation: Optional[str] = None,
        **kwargs
    ) -> None:
        self.random_seed = kwargs.pop('random_seed', None)
        super().__init__(**kwargs)
        self._model = model
        self._activation = keras.activations.get(output_activation)
        self._built = False

    def __init__(
        self,
        model: keras.Model,
        output_activation: Optional[str] = None,
        **kwargs
    ) -> None:
        self.random_seed = kwargs.pop('random_seed', None)
        super().__init__(**kwargs)
        self._model = model
        self._activation = keras.activations.get(output_activation)
        self._built = False

    def compute_saliency(
        self, 
        x: GraphTensor, 
        y: Optional[tf.Tensor],
    ) -> tf.Tensor:
        gradients = self.compute_gradients(x, y)
        gradients = tf.abs(gradients)
        return tf.reduce_sum(gradients, axis=1)
    
    def compute_gradients(
        self, 
        x: GraphTensor, 
        y: Optional[tf.Tensor],
    ) -> tf.Tensor:
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x.node_feature)
            predictions = self._model(x)
            predictions = self._activation(predictions)
            predictions = self._process_predictions(predictions, y)
        return tape.gradient(predictions, x.node_feature)

    @automatically_infer_input_signature
    def __call__(
        self, 
        x: GraphTensor, 
        y: Optional[tf.Tensor] = None
    ) -> Union[tf.Tensor, tf.RaggedTensor]:
        tf.random.set_seed(self.random_seed)
        x_orig = x
        if isinstance(x.node_feature, tf.RaggedTensor):
            x = x.merge()
        saliency = self.compute_saliency(x, y)
        return x_orig.update(node_feature=saliency).node_feature

    def build(
        self, 
        x: GraphTensor, 
        y: Optional[tf.Tensor],
    ) -> None:

        input_signature = [x.spec]
        if y is not None:
            y_signature = tf.TensorSpec(
                shape=tf.TensorShape([None]).concatenate(y.shape[1:]), 
                dtype=y.dtype)
            input_signature.append(y_signature)
            
        self.__call__ = tf.function(
            self.__call__, input_signature=input_signature)         
        
        self._built = True

    @staticmethod
    def _process_predictions(
        pred: tf.Tensor, 
        y: Optional[tf.Tensor],
    ) -> tf.Tensor:
        'Helper method to extract relevant predictions.'

        if y is None:
            return pred
        
        y = tf.cond(
            tf.rank(y) < 2,
            lambda: tf.expand_dims(y, -1),
            lambda: y)
        
        # If multi-label or multi-class, extract relevant preds.
        pred = tf.cond(
            tf.shape(y)[-1] > 1,
            lambda: tf.gather_nd(
                pred,
                tf.stack([
                    tf.range(tf.shape(y)[0], dtype=tf.int64),
                    tf.argmax(y, axis=-1)
                ], axis=1)
            ),
            lambda: pred)

        return pred


class IntegratedSaliencyMapping(SaliencyMapping):
    '''Integrated saliency mapping.

    Alias: ``IntegratedSaliency``

    Example usage:

    >>> encoder = molgraph.chemistry.MolecularGraphEncoder(
    ...     atom_encoder=molgraph.chemistry.Featurizer([
    ...         molgraph.chemistry.features.Symbol(),
    ...         molgraph.chemistry.features.Hybridization()
    ...     ])
    ... )
    >>> bbbp = molgraph.chemistry.datasets.get('bbbp')
    >>> bbbp['train']['x'] = encoder(bbbp['train']['x'])
    >>> bbbp['test']['x'] = encoder(bbbp['test']['x'])
    ...
    >>> gnn_model = tf.keras.Sequential([
    ...     molgraph.layers.GCNConv(units=128, name='gcn_conv_1'),
    ...     molgraph.layers.GCNConv(units=128, name='gcn_conv_2'),
    ...     molgraph.layers.GCNConv(units=128, name='gcn_conv_3'),
    ...     molgraph.layers.Readout('mean'),
    ...     tf.keras.layers.Dense(units=512),
    ...     tf.keras.layers.Dense(units=1, activation='sigmoid')
    ... ])
    ...
    >>> gnn_model.compile(optimizer='adam', loss='mse')
    >>> _ = gnn_model.fit(
    ...     bbbp['train']['x'], bbbp['train']['y'], epochs=10, verbose=0)
    ...
    >>> saliency = molgraph.models.IntegratedSaliencyMapping(model=gnn_model)
    >>> saliency_maps = saliency(bbbp['test']['x'].separate())
    '''

    def __init__(
        self,
        model: keras.Model,
        output_activation: Union[
            None, str, Callable[[tf.Tensor], tf.Tensor]] = None,
        steps: int = 20,
        **kwargs,
    ) -> None:
        super().__init__(
            model=model,
            output_activation=output_activation,
            **kwargs)
        self.steps = steps

    def compute_saliency(
        self, 
        x: GraphTensor, 
        y: Optional[tf.Tensor],
    ) -> tf.Tensor:

        original = x.node_feature
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
        cond = lambda x, gradients_batch, i: tf.less(i, self.steps + 1)
        x, gradients_batch, i = tf.while_loop(
            cond=cond, 
            body=body, 
            loop_vars=[x, gradients_batch, i])

        gradients_batch = gradients_batch.stack()

        integrated_gradients = (gradients_batch[:-1] + gradients_batch[1:]) / 2.0
        integrated_gradients = tf.math.reduce_mean(integrated_gradients, axis=0)

        integrated_gradients = (original - baseline) * integrated_gradients

        return tf.reduce_sum(tf.abs(integrated_gradients), axis=1)


class SmoothGradSaliencyMapping(SaliencyMapping):
    '''Smooth-gradient saliency mapping.

    Alias: ``SmoothGradSaliency``

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
    >>> saliency = molgraph.models.SmoothGradSaliencyMapping(model=gnn_model)
    >>> saliency_maps = saliency(esol['test']['x'].separate())
    '''
    def __init__(
        self,
        model: keras.Model,
        output_activation: Union[
            None, str, Callable[[tf.Tensor], tf.Tensor]] = None,
        steps: int = 50,
        noise: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(
            model=model,
            output_activation=output_activation,
            **kwargs)
        self.steps = steps
        self.noise = noise

    def compute_saliency(
        self, 
        x: GraphTensor, 
        y: Optional[tf.Tensor],
    ) -> tf.Tensor:

        original = x.node_feature

        gradients_batch = tf.TensorArray(original.dtype, size=self.steps)

        def body(x, gradients_batch, i):
            noisy = original + tf.random.normal(
                shape=(1, tf.shape(original)[1]),
                mean=0.0,
                stddev=self.noise,
                dtype=original.dtype,
                seed=self.random_seed,
            )
            x = x.update({'node_feature': noisy})
            gradients = self.compute_gradients(x, y)
            gradients_batch = gradients_batch.write(i, tf.abs(gradients))
            return x, gradients_batch, tf.add(i, 1)

        i = tf.constant(0)
        cond = lambda x, gradients_batch, i: tf.less(i, self.steps)
        x, gradients_batch, i = tf.while_loop(
            cond=cond, 
            body=body, 
            loop_vars=[x, gradients_batch, i])

        gradients_batch = gradients_batch.stack()

        gradients_average = tf.math.reduce_mean(gradients_batch, axis=0)

        return tf.reduce_sum(gradients_average, axis=1)

