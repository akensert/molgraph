import tensorflow as tf
from tensorflow import keras

from molgraph.internal import register_keras_serializable 

from molgraph.tensors.graph_tensor import GraphTensor


@register_keras_serializable(package='molgraph')
class FeatureMasking(keras.layers.Layer):
    '''Randomly masking node or edge features from the graph.

    Important: Requires node (or edge) features to be tokenized. I.e., requires 
    the `GraphTensor` instance to be produced from `molgraph.chemistry.Tokenizer` 
    instead of `molgraph.chemistry.Tokenizer`.

    Instead of specifying `feature`, ``NodeFeatureMasking(...)`` or 
    ``EdgeFeatureMasking(...)`` can be used instead.

    **Example:**

    >>> graph_tensor = molgraph.GraphTensor(
    ...     sizes=[5],
    ...     edge_src=[1, 4, 0, 2, 3, 1, 1, 0],
    ...     edge_dst=[0, 0, 1, 1, 1, 2, 3, 4],
    ...     node_feature=['Sym:C|Hyb:SP3', 'Sym:C|Hyb:SP2', 'Sym:O|Hyb:SP2',
    ...                   'Sym:O|Hyb:SP2', 'Sym:N|Hyb:SP3'],
    ...     edge_feature=['BonTyp:SINGLE|Rot:1', 'BonTyp:SINGLE|Rot:0',
    ...                   'BonTyp:SINGLE|Rot:1', 'BonTyp:DOUBLE|Rot:0',
    ...                   'BonTyp:SINGLE|Rot:0', 'BonTyp:DOUBLE|Rot:0',
    ...                   'BonTyp:SINGLE|Rot:0', 'BonTyp:SINGLE|Rot:0'],
    ... )
    >>> node_embedding = molgraph.layers.NodeEmbeddingLookup(
    ...    32, mask_token='[MASK]')
    >>> edge_embedding = molgraph.layers.EdgeEmbeddingLookup(
    ...    32, mask_token='[MASK]')
    >>> node_embedding.adapt(graph_tensor)
    >>> edge_embedding.adapt(graph_tensor)
    >>> model = tf.keras.Sequential([
    ...     molgraph.layers.NodeFeatureMasking(rate=0.15, mask_token='[MASK]'),
    ...     node_embedding,
    ...     molgraph.layers.EdgeFeatureMasking(rate=0.15, mask_token='[MASK]'),
    ...     edge_embedding,
    ... ])
    >>> output = model(graph_tensor)
    >>> output.node_feature.shape, output.edge_feature.shape
    (TensorShape([5, 32]), TensorShape([8, 32]))

    Args:
        rate (float):
            The dropout rate. Default to 0.15.
        mask_token (str):
            The mask token. Default to '[MASK]'.
        **kwargs:
            Specify the relevant ``feature``. Default to ``node_feature``.
            The reminaing kwargs are passed to the parent class.
    '''

    def __init__(
        self, 
        rate: float = 0.15, 
        mask_token: str = '[MASK]', 
        **kwargs
    ):
        if 'feature' in kwargs:
            self.feature = kwargs.pop('feature')
        elif not hasattr(self, 'feature'):
            self.feature = 'node_feature'
        super().__init__(**kwargs)
        self.rate = rate
        self.mask_token = mask_token

    def call(self, tensor: GraphTensor) -> GraphTensor:
        '''Defines the computation from inputs to outputs.

        This method should not be called directly, but indirectly
        via ``__call__()``. Upon first call, the layer is automatically
        built via ``build()``.

        Args:
            tensor (GraphTensor):
                Input to the layer.

        Returns:
            GraphTensor: a masked instance of GraphTensor.
        '''
        feature = getattr(tensor, self.feature)

        if isinstance(feature, tf.RaggedTensor):
            feature = feature.flat_values

        mask = tf.random.uniform(tf.shape(feature)[:1]) <= self.rate
        feature = tf.where(mask, self.mask_token, feature)
        return tensor.update({self.feature: feature})

    def get_config(self):
        config = {
            'feature': self.feature, 
            'rate': self.rate, 
            'mask_token': self.mask_token
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@register_keras_serializable(package='molgraph')
class NodeFeatureMasking(FeatureMasking):
    feature = 'node_feature'


@register_keras_serializable(package='molgraph')
class EdgeFeatureMasking(FeatureMasking):
    feature = 'edge_feature'
