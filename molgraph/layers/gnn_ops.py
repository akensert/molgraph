import tensorflow as tf
from tensorflow import keras

from typing import Optional
from typing import Tuple


def propagate_node_features(
    *,
    node_feature: tf.Tensor,
    edge_src: tf.Tensor,
    edge_dst: tf.Tensor,
    edge_weight: Optional[tf.Tensor] = None,
    mode: str = 'sum'
) -> tf.Tensor:
    '''Aggregates source node features to destination node features.

    Args:
        node_feature (tf.Tensor):
            Node features of `GraphTensor` instance.
        edge_src (tf.Tensor):
            Source node indices of `GraphTensor` instance. Entry i 
            corresponds to row i in `node_feature`.
        edge_dst (tf.Tensor):
            Destination node indices of `GraphTensor` instance. Entry i 
            corresponds to row i in `node_feature`.
        edge_weight (tf.Tensor):
            Edge weights associated with `edge_dst` and `edge_src`.
        mode (str):
            The type of aggregation to be performed, either of 'sum', 'mean',
            'min' or 'max'. If None, 'sum' will be used. Default to 'sum'.

    Returns:
        tf.Tensor: Updated node features.
    '''
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

def softmax_edge_weights(
    *,
    edge_weight: tf.Tensor,
    edge_dst: tf.Tensor,
    exponentiate: bool = True,
    clip_values: Tuple[float, float] = (-5.0, 5.0),
) -> tf.Tensor:
    '''Normalizes edge weights via softmax.

    The sum of edge weights associated with each destination node sums to 1.

    Args:
        edge_weight (tf.Tensor):
            Edge weights associated with `edge_src` and `edge_dst` of the
            `GraphTensor` instance.
        edge_dst (tf.Tensor):
            Destination node indices (corresponding to `edge_weight`) of the
            `GraphTensor` instance.
        exponentiate (bool):
            Whether `edge_weight` should be exponentiated. Default to True.
        clip_values (tuple):
            If exponentiation, clip values before it, for stability.

    Returns:
        tf.Tensor: Normalized edge weights.
    '''

    def true_fn(edge_weight, edge_dst, exponentiate, clip_values):
        'If edges exist, call this function.'
        if exponentiate:
            edge_weight = tf.clip_by_value(edge_weight, *clip_values)
            edge_weight = tf.math.exp(edge_weight)
        num_segments = tf.maximum(tf.reduce_max(edge_dst) + 1, 1)
        edge_weight_sum = tf.math.unsorted_segment_sum(
            edge_weight, edge_dst, num_segments) + keras.backend.epsilon()
        edge_weight_sum = tf.gather(edge_weight_sum, edge_dst)
        return edge_weight / edge_weight_sum

    def false_fn(edge_weight):
        'If no edges exist, call this function.'
        return edge_weight

    return tf.cond(
        tf.greater(tf.shape(edge_dst)[0], 0),
        lambda: true_fn(edge_weight, edge_dst, exponentiate, clip_values),
        lambda: false_fn(edge_weight))

def compute_edge_weights_from_degrees(
    *,
    edge_src: tf.Tensor,
    edge_dst: tf.Tensor,
    edge_feature: Optional[tf.Tensor] = None,
    mode: Optional[str] = 'symmetric',
) -> tf.Tensor:
    '''Computes edge weights based on the number of nearest neighbors.

    These edge weights can subsequently be used as normalization coefficients
    for the neighborhood aggregation (``propagate_node_features``).

    Args:
        edge_src (tf.Tensor):
            Source node indices.
        edge_dst (tf.Tensor):
            Destination node indices.
        edge_feature (tf.Tensor):
            Edge features associated with edge_dst and edge_src.
        mode (str):
            Type of normalization to use. Either of 'symmetric', 'row' or None.
            If None, 'row' is used. Default to 'symmetric'.
    Returns:
        tf.Tensor: Edge weights based on degrees.
    '''

    if edge_feature is None:
        edge_feature = tf.ones((tf.shape(edge_dst)[0], 1), dtype=tf.float32)
    else:
        edge_feature = tf.expand_dims(edge_feature, axis=1)

    def true_fn(edge_src, edge_dst, edge_feature, mode):
        'If edges exist, call this function.'
        # divide_no_nan makes sense? or tf.where(deg) -> .. * edge_feature
        degree = tf.math.unsorted_segment_sum(
            edge_feature, edge_dst, tf.reduce_max(edge_dst) + 1)

        if mode == 'symmetric':
            # symmetric norm (in matrix notation: A' = D^{-0.5} @ A @ D^{-0.5})
            adjacency = tf.stack([edge_src, edge_dst], axis=1)
            degree = tf.sqrt(tf.reduce_prod(tf.gather(degree, adjacency), 1))
            edge_weight = tf.math.divide_no_nan(1., degree)
        else:
            # row norm (in matrix notation: A' = D^{-1} @ A)
            degree = tf.gather(degree, edge_dst)
            edge_weight = tf.math.divide_no_nan(1., degree)

        return edge_weight

    def false_fn(edge_feature):
        'If no edges exist, call this function.'
        edge_weight = tf.zeros(
            tf.TensorShape([0]).concatenate(edge_feature.shape[1:]),
            dtype=tf.float32)
        return edge_weight

    return tf.cond(
        tf.greater(tf.shape(edge_src)[0], 0),
        lambda: true_fn(edge_src, edge_dst, edge_feature, mode),
        lambda: false_fn(edge_feature)
    )

def reduce_features(
    *,
    feature: tf.Tensor,
    mode: Optional[str],
    output_units: Optional[int] = None,
    axis: int = 1,
) -> tf.Tensor:
    '''Reduces dimension of features. 
    
    Useful to concatenate, sum or average heads of GATConv or GTConv.

    Args:
        feature (tf.Tensor):
            The features to be reduced. Either node or edge features.
        mode (str, None):
            The type of reduction to be performed. Either of 'concat', 
            'sum', 'mean' or None. If None, 'mean' is performed.
        output_units (int, None):
            The output dimension (innermost dimension) after reshaping. Only
            relevant if ``mode='concat'``. If None, dim is inferred from the
            two innermost dimensions. Default to None,
        axis (int):
            Axis to reduce. Default to 1.

    Returns:
        tf.Tensor: Reduced features (rank 3 -> rank 2).
    '''
    if mode == 'concat':
        if output_units is None:
            output_units = tf.reduce_prod(feature.shape[1:])
        return tf.reshape(feature, (-1, output_units))
    elif mode == 'sum':
        return tf.reduce_sum(feature, axis=axis)
    return tf.reduce_mean(feature, axis=axis)
