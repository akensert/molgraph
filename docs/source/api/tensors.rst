Graph Tensor
============

**Code snippet:**

.. code-block::

  from tensorflow.random import uniform
  from molgraph.tensors import GraphTensor

  # Example 1: dict of ragged arrays
  graph_tensor = GraphTensor(
      data={
          'edge_dst': [[0, 1], [0, 0, 1, 1, 2, 2]],
          'edge_src': [[1, 0], [1, 2, 0, 2, 1, 0]],
          'node_feature': [
              [[1.0, 0.0], [1.0, 0.0]],
              [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
          ],
      }
  )

  # Example 2: dict of arrays
  graph_tensor = GraphTensor(
      data={
          'edge_dst': [[0, 1, 2, 2, 3, 3, 4, 4]],
          'edge_src': [[1, 0, 3, 4, 2, 4, 3, 2]],
          'node_feature': [
              [1.0, 0.0],
              [1.0, 0.0],
              [1.0, 0.0],
              [1.0, 0.0],
              [0.0, 1.0]
          ],
      }
  )

  # Example 3: keyword arguments
  graph_tensor = GraphTensor(
      edge_dst=[[0, 1], [0, 0, 1, 1, 2, 2]],
      edge_src=[[1, 0], [1, 2, 0, 2, 1, 0]],
      node_feature=[
          [[1.0, 0.0], [1.0, 0.0]],
          [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
      ]
  )

  # Merge subgraphs
  graph_tensor = graph_tensor.merge()

  # Add and then remove random feature
  graph_tensor = graph_tensor.update({
      'random_feature': uniform(graph_tensor['node_feature'].shape)})

  graph_tensor = graph_tensor.remove(['random_feature'])

  # Separate subgraphs
  graph_tensor = graph_tensor.separate()

.. autoclass:: molgraph.tensors.GraphTensor(tensorflow.python.framework.composite_tensor.CompositeTensor)
  :undoc-members:
  :members:
  :special-members: __init__, __getitem__, __getattr__
