Graph Tensor
============

.. autoclass:: molgraph.tensors.GraphTensor(tensorflow.python.framework.composite_tensor.CompositeTensor)
  :members: merge, separate, update, remove, propagate, is_ragged, spec, unspecific_spec, shape, dtype, rank
  :special-members: __getitem__, __getattr__, node_feature, edge_src, edge_dst, edge_feature, graph_indicator
  :member-order: bysource

.. autoclass:: molgraph.tensors.GraphTensorSpec(tensorflow.python.framework.type_spec.BatchableTypeSpec)
  :members: shape, dtype, rank, node_feature, edge_src, edge_dst, edge_feature, graph_indicator
  :member-order: bysource

