import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.keras import activations
from typing import Optional

from molgraph.tensors.graph_tensor import GraphTensor


@keras.utils.register_keras_serializable(package='molgraph')
class EmbeddingLookup(layers.StringLookup):

    def __init__(
        self,
        output_dim,
        input_dim: Optional[int] = None,
        embeddings_initializer='uniform',
        embeddings_regularizer=None,
        embeddings_constraint=None,
        max_tokens=None,
        num_oov_indices=1,
        mask_token='[MASK]',
        oov_token='[UNK]',
        vocabulary=None,
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
        if not isinstance(data,  GraphTensor):
            data = data.map(lambda x: getattr(x, self.feature))
        else:
            data = getattr(data, self.feature)
        super().adapt(data, batch_size=batch_size, steps=steps)
        self._vocabulary_size = self.vocabulary_size()

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

    def call(self, tensor: GraphTensor) -> GraphTensor:
        if not self._built_from_vocabulary_size:
            self._build_from_vocabulary_size(self._vocabulary_size)

        tensor = tensor.update({
            self.feature: super().call(getattr(tensor, self.feature))
        })
        return tensor.update({
            self.feature: tf.nn.embedding_lookup(
                self.embeddings, getattr(tensor, self.feature))
        })

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
