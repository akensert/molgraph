import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.keras import activations

from typing import Optional
from typing import Union
from typing import List

from molgraph.tensors.graph_tensor import GraphTensor


@keras.utils.register_keras_serializable(package='molgraph')
class EmbeddingLookup(layers.StringLookup):

    '''A loookup layer and embedding layer in combination.

    Specify, as keyword argument only,
    ``EmbeddingLookup(feature='node_feature')`` to perform standard scaling
    on the ``node_feature`` field of the ``GraphTensor``, or,
    ``EmbeddingLookup(feature='edge_feature')`` to perform standard scaling
    on the ``edge_feature`` field of the ``GraphTensor``. If not specified,
    the ``node_feature`` field will be considered.

    **Examples:**

    Adapt layer on ``GraphTensor`` directly:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     data={
    ...         'edge_dst': [0, 1, 2, 2, 3, 3, 4, 4],
    ...         'edge_src': [1, 0, 3, 4, 2, 4, 3, 2],
    ...         'node_feature': [
    ...             'Sym:C', 'Sym:C', 'Sym:C', 'Sym:O', 'Sym:N',
    ...         ],
    ...         'graph_indicator': [0, 0, 1, 1, 1],
    ...     }
    ... )
    >>> # Initialize layer
    >>> embedding = molgraph.layers.EmbeddingLookup(
    ...    feature='node_feature', output_dim=4)
    >>> # Adapt layer to graph_tensor
    >>> embedding.adapt(graph_tensor)
    >>> model = tf.keras.Sequential([
    ...     tf.keras.layers.Input(type_spec=graph_tensor.unspecific_spec),
    ...     embedding,
    ... ])
    >>> graph_tensor = model(graph_tensor)
    >>> graph_tensor.node_feature.shape
    TensorShape([5, 4])

    Adapt layer on dataset, constructed from ``GraphTensor``:

    >>> graph_tensor = molgraph.GraphTensor(
    ...     data={
    ...         'edge_dst': [[0, 1], [2, 2, 3, 3, 4, 4]],
    ...         'edge_src': [[1, 0], [3, 4, 2, 4, 3, 2]],
    ...         'node_feature': [
    ...             ['Sym:C', 'Sym:C'], ['Sym:C', 'Sym:O', 'Sym:N'],
    ...         ],
    ...     }
    ... )
    >>> # Obtain dataset
    >>> ds = tf.data.Dataset.from_tensor_slices(graph_tensor).batch(2)
    >>> # Initialize layer
    >>> embedding = molgraph.layers.EmbeddingLookup(
    ...    feature='node_feature', output_dim=4)
    >>> # Adapt layer to graph_tensor
    >>> embedding.adapt(ds)
    >>> # Build model
    >>> model = tf.keras.Sequential([
    ...     tf.keras.layers.Input(type_spec=graph_tensor.unspecific_spec),
    ...     embedding,
    ... ])
    >>> # Predict (obtain new GraphTensor)
    >>> output = model.predict(ds)
    >>> output.node_feature.shape
    TensorShape([2, None, 4])

    Args:
        output_dim (int):
            The output dimension of the embedding layer.
        input_dim (int, None):
            The input dimension to the embedding layer. If None, the input
            dimension is determined by the vocabulary size (obtained from
            adapting the StringLookup layer to the data). Default to None.
        embedding_initializer (tf.keras.initializers.Initializer, str):
            Initializer function for the embedding. Default to ``'uniform'``.
        embedding_regularizer (tf.keras.regularizers.Regularizer, None):
            Regularizer function applied to the embedding. Default to None.
        embedding_constraint (tf.keras.constraints.Constraint, None):
            Constraint function applied to the embedding. Default to None.
        max_tokens (int, None):
            Maximum number of tokens to use. Default to None.
        num_oov_indices (int):
            Number of out-of-vocabulary indices to use. Default to 1.
        mask_token (str):
            The token that represents masked input. Default to ``'[MASK]'``.
        oov_token (str):
            The token that represents out-of-vocabulary input. Default to ``'[UNK]'``.
        vocabulary (list, None):
            Optional vocabulary. If None, obtain a vocabulary via the ``adapt()``
            method. Default to None.
        **kwargs:
            Specify the relevant ``feature``. Default to ``node_feature``.
            The reminaing kwargs are passed to the parent class.
    '''

    def __init__(
        self,
        output_dim,
        input_dim: Optional[int] = None,
        embeddings_initializer: Union[str, initializers.Initializer] = 'uniform',
        embeddings_regularizer: Optional[regularizers.Regularizer] = None,
        embeddings_constraint: Optional[constraints.Constraint] = None,
        max_tokens: Optional[int] = None,
        num_oov_indices: int = 1,
        mask_token: str = '[MASK]',
        oov_token: str = '[UNK]',
        vocabulary: Optional[List[str]] = None,
        **kwargs
    ):
        if 'feature' in kwargs:
            self.feature = kwargs['feature']
            del kwargs['feature']
        elif not hasattr(self, 'feature'):
            self.feature = 'node_feature'

        super().__init__(
            max_tokens=max_tokens,
            num_oov_indices=num_oov_indices,
            mask_token=mask_token,
            oov_token=oov_token,
            vocabulary=vocabulary,
            **kwargs
        )

        self.output_dim = output_dim
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.embeddings_constraint = constraints.get(embeddings_constraint)
        self._vocabulary_size = None
        self._built_from_vocabulary_size = False

    def adapt(self, data, batch_size=None, steps=None):
        '''Adapts the layer to data.

        When adapting the layer to the data, ``build()`` will be called
        automatically (to initialize the relevant attributes). After adaption,
        the layer is finalized and ready to be used.

        Args:
            data (GraphTensor, tf.data.Dataset):
                Data to be used to adapt the layer. Can be either a
                ``GraphTensor`` directly or a ``tf.data.Dataset`` constructed
                from a ``GraphTensor``.
            batch_size (int, None):
                The batch size to be used during adaption. Default to None.
            steps (int, None):
                The number of steps of adaption. If None, the number of
                samples divided by the batch_size is used. Default to None.
        '''
        if not isinstance(data,  GraphTensor):
            data = data = data.map(
                lambda x: getattr(x, self.feature))
        else:
            data = getattr(data, self.feature)
        super().adapt(data, batch_size=batch_size, steps=steps)
        self._vocabulary_size = self.vocabulary_size()

    def call(self, tensor: GraphTensor) -> GraphTensor:
        '''Defines the computation from inputs to outputs.

        This method should not be called directly, but indirectly
        via ``__call__()``. Upon first call, the layer is automatically
        built via ``build()``.

        Args:
            data (GraphTensor):
                Input to the layer.

        Returns:
            GraphTensor:
                A ``GraphTensor`` with updated features. Either the
                ``node_features`` field or the ``edge_features``
                field (of the ``GraphTensor``) are updated.
        '''
        if not self._built_from_vocabulary_size:
            self._build_from_vocabulary_size(self._vocabulary_size)

        tensor = tensor.update({
            self.feature: super().call(getattr(tensor, self.feature))
        })
        return tensor.update({
            self.feature: tf.nn.embedding_lookup(
                self.embeddings, getattr(tensor, self.feature))
        })

    def _build_from_vocabulary_size(self, vocabulary_size):
        self._built_from_vocabulary_size = True

        self.embeddings = self.add_weight(
            shape=(vocabulary_size, self.output_dim),
            dtype=tf.float32,
            initializer=self.embeddings_initializer,
            name='embeddings',
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            experimental_autocast=False
        )

    @classmethod
    def from_config(cls, config):
        vocabulary_size = config.pop('vocabulary_size')
        layer = cls(**config)
        if vocabulary_size is None:
            pass # TODO(akensert): add warning message about not restoring weights
        else:
            layer._build_from_vocabulary_size(vocabulary_size)
        return layer

    def get_config(self):
        base_config = super().get_config()
        base_config.update({
            'vocabulary_size': self._vocabulary_size,
            'output_dim': self.output_dim,
            'embeddings_initializer': initializers.serialize(
                self.embeddings_initializer),
            'embeddings_regularizer': regularizers.serialize(
                self.embeddings_regularizer),
            'embeddings_constraint': constraints.serialize(
                self.embeddings_constraint),
        })
        return base_config


@keras.utils.register_keras_serializable(package='molgraph')
class NodeEmbeddingLookup(EmbeddingLookup):
    feature = 'node_feature'


@keras.utils.register_keras_serializable(package='molgraph')
class EdgeEmbeddingLookup(EmbeddingLookup):
    feature = 'edge_feature'
