import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from molgraph.tensors.graph_tensor import GraphTensor



@keras.utils.register_keras_serializable(package='molgraph')
class SegmentPoolingReadout(layers.Layer):

    '''Segmentation pooling for graph readout.

    Args:
        mode (str):
            What type of pooling should be performed. Either of 'mean',
            'max' or 'sum'. Specifically, performs a ``tf.math.segment_mean``,
            ``tf.math.segment_max`` or ``tf.math.segment_sum`` (respectively)
            on the node_feature component of the inputted graph tensor.
            Defaults to 'mean'.
    '''

    def __init__(self, mode: str = 'mean', **kwargs) -> None:
        super().__init__(**kwargs)
        self.mode = mode
        if self.mode == 'mean' or self.mode == 'average' or self.mode == 'avg':
            self.pooling_fn = tf.math.segment_mean
        elif self.mode == 'sum':
            self.pooling_fn = tf.math.segment_sum
        elif self.mode == 'max':
            self.pooling_fn = tf.math.segment_max
        else:
            raise ValueError('Value passed to mode is invalid, ' +
                             'needs to be one of the following: ' +
                             '"mean"/"average"/"avg", "sum" or "max"')

    def call(self, tensor: GraphTensor) -> tf.Tensor:
        '''Defines the computation from inputs to outputs.

        This method should not be called directly, but indirectly
        via ``__call__()``. Upon first call, the layer is automatically
        built via ``build()``.

        Args:
            tensor (GraphTensor):
                A graph tensor which serves as input to the layer.

        Returns:
            tf.Tensor:
                A tensor based on the node_feature component of the inputted
                graph tensor.
        '''
        if isinstance(tensor.node_feature, tf.RaggedTensor):
            tensor = tensor.merge()
        node_feature = self.pooling_fn(
            tensor.node_feature, tensor.graph_indicator)
        return node_feature

    def get_config(self):
        config = super().get_config()
        config.update({'mode': self.mode})
        return config
