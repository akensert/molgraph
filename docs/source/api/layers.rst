###########################
Layers
###########################

***************************
GNN Layer
***************************

.. autoclass:: molgraph.layers.GNNLayer(tf.keras.layers.Layer)
  :members: _call, _build, call, build_from_signature, propagate, 
            compute_output_shape, compute_output_signature, 
            get_dense, get_einsum_dense, get_kernel, get_bias

***************************
GNN ops
***************************

GNN ops are helper functions which makes it easier to code up a custom
GNN layer. For example, a basic GCN layer can be coded up as follows:

.. code-block::

  import tensorflow as tf
  from molgraph.layers import gnn_ops

  class MyGCNConv(tf.keras.layers.Layer):

    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.units),
            dtype=tf.float32,
            trainable=True)
        self.built = True

    def call(self, graph_tensor):
        graph_tensor_orig = graph_tensor
        if isinstance(graph_tensor.node_feature, tf.RaggedTensor):
            graph_tensor = graph_tensor.merge()
        node_feature_transformed = tf.matmul(graph_tensor.node_feature, self.kernel)
        node_feature_aggregated = gnn_ops.propagate_node_features(
            node_feature=node_feature_transformed,
            edge_dst=graph_tensor.edge_dst,
            edge_src=graph_tensor.edge_src,
            mode='mean')
        return graph_tensor_orig.update({'node_feature': node_feature_aggregated})


.. automodule:: molgraph.layers.gnn_ops
  :members:
  :member-order: bysource

***************************
Convolutional
***************************

GCNConv
===========================
.. autoclass:: molgraph.layers.GCNConv(molgraph.layers.GNNLayer)
  :members: call, get_config, from_config, compute_output_shape

GINConv
===========================
.. autoclass:: molgraph.layers.GINConv(molgraph.layers.GNNLayer)
  :members: call, get_config, from_config, compute_output_shape

GraphSageConv
===========================
.. autoclass:: molgraph.layers.GraphSageConv(molgraph.layers.GNNLayer)
  :members: call, get_config, from_config, compute_output_shape

GCNIIConv
===========================
.. autoclass:: molgraph.layers.GCNIIConv(molgraph.layers.GNNLayer)
  :members: call, get_config, from_config, compute_output_shape


***************************
Attentional
***************************

GATConv
===========================
.. autoclass:: molgraph.layers.GATConv(molgraph.layers.GNNLayer)
  :members: call, get_config, from_config, compute_output_shape

GATv2Conv
===========================
.. autoclass:: molgraph.layers.GATv2Conv(molgraph.layers.GNNLayer)
  :members: call, get_config, from_config, compute_output_shape

GatedGCNConv
===========================
.. autoclass:: molgraph.layers.GatedGCNConv(molgraph.layers.GNNLayer)
  :members: call, get_config, from_config, compute_output_shape

GMMConv
===========================
.. autoclass:: molgraph.layers.GMMConv(molgraph.layers.GNNLayer)
  :members: call, get_config, from_config, compute_output_shape

GTConv
===========================
.. autoclass:: molgraph.layers.GTConv(molgraph.layers.GNNLayer)
  :members: call, get_config, from_config, compute_output_shape

AttentiveFPConv
===========================
.. autoclass:: molgraph.layers.AttentiveFPConv(molgraph.layers.GATConv)
  :members: call, build_from_signature, get_config, from_config, 
            compute_output_shape, compute_output_signature


***************************
Message-passing
***************************

MPNNConv
===========================
  .. autoclass:: molgraph.layers.MPNNConv(molgraph.layers.GNNLayer)
    :members: call, get_config, from_config, compute_output_shape

EdgeConv
===========================
  .. autoclass:: molgraph.layers.EdgeConv(tf.keras.layers.Layer)
    :members: call, get_config, from_config, compute_output_shape


***************************
Geometric
***************************

DTNNConv
===========================
.. autoclass:: molgraph.layers.DTNNConv(molgraph.layers.GNNLayer)
  :members: call, get_config, from_config, compute_output_shape

GCFConv
===========================
.. autoclass:: molgraph.layers.GCFConv(molgraph.layers.GNNLayer)
  :members: call, get_config, from_config, compute_output_shape


***************************
Readout
***************************

SegmentPoolingReadout
===========================
.. autoclass:: molgraph.layers.SegmentPoolingReadout(tensorflow.keras.layers.Layer)
  :members: call, get_config, from_config, compute_output_shape

TransformerEncoderReadout
===========================
.. autoclass:: molgraph.layers.TransformerEncoderReadout(tensorflow.keras.layers.Layer)
  :members: call, get_config, from_config, compute_output_shape

SetGatherReadout
===========================
.. autoclass:: molgraph.layers.SetGatherReadout(tensorflow.keras.layers.Layer)
  :members: call, get_config, from_config, compute_output_shape

NodeReadout
===========================
.. autoclass:: molgraph.layers.NodeReadout(tensorflow.keras.layers.Layer)
  :members: call, get_config, from_config, compute_output_shape

AttentiveFPReadout
===========================
.. autoclass:: molgraph.layers.AttentiveFPReadout(tensorflow.keras.layers.Layer)
  :members: call, get_config, from_config, compute_output_shape



***************************
Positional encoding
***************************

LaplacianPositionalEncodig
===========================
.. autoclass:: molgraph.layers.LaplacianPositionalEncoding(tensorflow.keras.layers.Layer)
  :members: call, get_config, from_config, compute_output_shape


***************************
Preprocessing
***************************

MinMaxScaling
===========================
.. autoclass:: molgraph.layers.MinMaxScaling(layers.experimental.preprocessing.PreprocessingLayer)
  :members: call, adapt, get_config, from_config
  :member-order: bysource

StandardScaling
===========================
.. autoclass:: molgraph.layers.StandardScaling(layers.experimental.preprocessing.PreprocessingLayer)
  :members: call, adapt, get_config, from_config
  :member-order: bysource

VarianceThreshold
===========================
.. autoclass:: molgraph.layers.VarianceThreshold(molgraph.layers.StandardScaling)
  :members: call, adapt, get_config, from_config
  :member-order: bysource

CenterScaling
===========================
.. autoclass:: molgraph.layers.CenterScaling(layers.experimental.preprocessing.PreprocessingLayer)
  :members: call, adapt, get_config, from_config
  :member-order: bysource

EmbeddingLookup
===========================
.. autoclass:: molgraph.layers.EmbeddingLookup(tensorflow.keras.layers.StringLookup)
  :members: call, adapt, get_config, from_config
  :member-order: bysource

FeatureProjection
===========================
.. autoclass:: molgraph.layers.FeatureProjection(tensorflow.keras.layers.Layer)
  :members: get_config, from_config, call
  :member-order: bysource

FeatureMasking
===========================
.. autoclass:: molgraph.layers.FeatureMasking(tensorflow.keras.layers.Layer)
  :members: get_config, from_config, call
  :member-order: bysource

NodeDropout
===========================
.. autoclass:: molgraph.layers.NodeDropout(tensorflow.keras.layers.Layer)
  :members: get_config, from_config, call
  :member-order: bysource

EdgeDropout
===========================
.. autoclass:: molgraph.layers.EdgeDropout(tensorflow.keras.layers.Layer)
  :members: get_config, from_config, call
  :member-order: bysource


***************************
Postprocessing
***************************

DotProductIncident
===========================
.. autoclass:: molgraph.layers.DotProductIncident(tensorflow.keras.layers.Layer)
  :members: call, get_config, from_config,
  :member-order: bysource

GatherIncident
===========================
.. autoclass:: molgraph.layers.GatherIncident(tensorflow.keras.layers.Layer)
  :members: call, get_config, from_config,
  :member-order: bysource

ExtractField
===========================
.. autoclass:: molgraph.layers.ExtractField(tensorflow.keras.layers.Layer)
  :members: call, get_config, from_config,
  :member-order: bysource

