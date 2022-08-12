import tensorflow as tf
from tensorflow import keras

from typing import Optional
from typing import Tuple


def softmax_edge_weights(
    edge_weight: tf.Tensor,
    edge_dst: tf.Tensor,
    exponentiate: bool = True,
    clip_values: Tuple[float, float] = (-5.0, 5.0),
) -> tf.Tensor:

    def true_fn(edge_weight, edge_dst, exponentiate, clip_values):
        """If edges exist, call this function"""
        if exponentiate:
            edge_weight = tf.clip_by_value(edge_weight, *clip_values)
            edge_weight = tf.math.exp(edge_weight)
        num_segments = tf.maximum(tf.reduce_max(edge_dst) + 1, 1)
        edge_weight_sum = tf.math.unsorted_segment_sum(
            edge_weight, edge_dst, num_segments) + keras.backend.epsilon()
        repeats = tf.math.bincount(tf.maximum(edge_dst, 0))
        edge_weight_sum = tf.repeat(edge_weight_sum, repeats, axis=0)
        return edge_weight / edge_weight_sum

    def false_fn(edge_weight):
        """If no edges exist, call this function"""
        return edge_weight

    return tf.cond(
        tf.greater(tf.shape(edge_dst)[0], 0),
        lambda: true_fn(edge_weight, edge_dst, exponentiate, clip_values),
        lambda: false_fn(edge_weight))

def reduce_features(
    feature: tf.Tensor,
    mode: Optional[str],
    reduce_dim: int,
) -> tf.Tensor:
    if mode == 'concat':
        return tf.reshape(feature, (-1, reduce_dim))
    elif mode == 'sum':
        return tf.reduce_sum(feature, axis=1)
    return tf.reduce_mean(feature, axis=1)

def propagate_node_features(
    node_feature: tf.Tensor,
    edge_dst: tf.Tensor,
    edge_src: tf.Tensor,
    edge_weight: Optional[tf.Tensor] = None,
    mode: str = 'sum'
) -> tf.Tensor:
    """Aggregated neighboring node information (source nodes) to destination
    nodes. Thus, is edge depedent.
    """
    num_segments = tf.shape(node_feature)[0]
    node_feature = tf.gather(node_feature, edge_src)
    if edge_weight is not None:
        node_feature *= edge_weight
    node_feature = getattr(tf.math, f'unsorted_segment_{mode}')(
        node_feature, edge_dst, num_segments)
    # If no segment_id for data, unsorted_segment_max/min returns
    # smallest/largest value for the dtype (e.g., for 'float16',
    # -65500./65500. respectively)
    if mode == 'max':
        node_feature = tf.where(node_feature <= -65500., 0., node_feature)
    elif mode == 'min':
        node_feature = tf.where(node_feature >= 65500., 0., node_feature)
    return node_feature

def compute_edge_weights_from_degrees(
    edge_dst: tf.Tensor,
    edge_src: tf.Tensor,
    edge_feature: Optional[tf.Tensor] = None,
    mode: Optional[str] = 'symmetric',
) -> tf.Tensor:

    if edge_feature is None:
        edge_feature = tf.ones((tf.shape(edge_dst)[0], 1), dtype=tf.float32)
    else:
        edge_feature = tf.expand_dims(edge_feature, axis=1)

    def true_fn(edge_dst, edge_src, edge_feature, mode):
        """If edges exist, call this function"""
        # divide_no_nan makes sense? or tf.where(deg) -> .. * edge_feature
        degree = tf.math.unsorted_segment_sum(
            edge_feature, edge_dst, tf.reduce_max(edge_dst) + 1)

        if mode == 'symmetric':
            # symmetric norm (in matrix notation: A' = D^{-0.5} @ A @ D^{-0.5})
            adjacency = tf.stack([edge_dst, edge_src], axis=1)
            degree = tf.sqrt(tf.reduce_prod(tf.gather(degree, adjacency), 1))
            edge_weight = tf.math.divide_no_nan(1., degree)
        else:
            # row norm (in matrix notation: A' = D^{-1} @ A)
            degree = tf.gather(degree, edge_dst)
            edge_weight = tf.math.divide_no_nan(1., degree)

        return edge_weight

    def false_fn(edge_feature):
        """If no edges exist, call this function"""
        edge_weight = tf.zeros(
            tf.TensorShape([0]).concatenate(edge_feature.shape[1:]),
            dtype=tf.float32)
        return edge_weight

    return tf.cond(
        tf.greater(tf.shape(edge_dst)[0], 0),
        lambda: true_fn(edge_dst, edge_src, edge_feature, mode),
        lambda: false_fn(edge_feature)
    )
