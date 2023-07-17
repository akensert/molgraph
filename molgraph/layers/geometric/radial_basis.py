import tensorflow as tf
from tensorflow import keras

from typing import Optional
from typing import Union

from molgraph.internal import register_keras_serializable 


@register_keras_serializable(package='molgraph')
class RadialBasis(keras.layers.Layer):

    def __init__(
        self,
        distance_min: float = -1.0,
        distance_max: float = 18.0,
        distance_granularity: float = 0.1,
        stddev: Optional[Union[float, str]] = 'auto',
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.distance_min = distance_min
        self.distance_max = distance_max
        self.distance_granularity = distance_granularity
        if not isinstance(stddev, float) or stddev == 'auto':
            self.stddev = self.distance_granularity
        else:
            self.stddev = stddev
        self.centers = tf.range(
            self.distance_min, self.distance_max, self.distance_granularity)
        self.centers = tf.expand_dims(self.centers, axis=0)

    def call(self, distances: tf.Tensor) -> tf.Tensor:
        # distances are "edge_feature"
        if distances.shape.ndims < 2:
            distances = tf.expand_dims(distances, axis=1)
        return tf.math.exp(
            -tf.square(distances - self.centers) / (2 * tf.square(self.stddev))
        )

    def get_config(self):
        base_config = super().get_config()
        base_config.update({
            'distance_min': self.distance_min,
            'distance_max': self.distance_max,
            'distance_granularity': self.distance_granularity,
            'stddev': self.stddev
        })
        return base_config
