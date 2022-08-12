Layers
======

**Code snippet:**

.. code-block::

  from molgraph import layers
  from molgraph import chemistry

  encoder = chemistry.MolecularGraphEncoder(
      atom_encoder=chemistry.AtomFeaturizer(
          chemistry.atom_features.unpack()
      )
  )

  bbbp = chemistry.datasets.get('bbbp')

  bbbp['train']['x'] = encoder(bbbp['train']['x'])
  bbbp['test']['x'] = encoder(bbbp['test']['x'])

  # Pass GraphTensor to model
  gnn_model = tf.keras.Sequential([
    tf.keras.layers.Input(type_spec=bbbp['train']['x'].spec),
    layers.GCNConv(units=128),
    layers.GCNConv(units=128),
    layers.GCNConv(units=128),
    layers.Readout('mean'),
    tf.keras.layers.Dense(units=512),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
  ])
  gnn_model.compile(optimizer='adam', loss='bce')
  gnn_model.fit(bbbp['train']['x'], bbbp['train']['y'], epochs=10)
  gnn_model.evaluate(bbbp['test']['x'], bbbp['test']['y'])
  predictions = gnn_model.predict(bbbp['test']['x'])


Convolutional
~~~~~~~~~~~~~~
.. autoclass:: molgraph.layers.GCNConv(molgraph.layers.BaseLayer)
  :special-members: __init__
  :undoc-members: __init__

.. autoclass:: molgraph.layers.GINConv(molgraph.layers.BaseLayer)
  :special-members: __init__
  :undoc-members: __init__

.. autoclass:: molgraph.layers.GraphSageConv(molgraph.layers.BaseLayer)
  :special-members: __init__
  :undoc-members: __init__

.. autoclass:: molgraph.layers.GCNIIConv(molgraph.layers.BaseLayer)
  :special-members: __init__
  :undoc-members: __init__


Attentional
~~~~~~~~~~~~~~
.. autoclass:: molgraph.layers.GATConv(molgraph.layers.BaseLayer)
  :special-members: __init__
  :undoc-members: __init__

.. autoclass:: molgraph.layers.GatedGCNConv(molgraph.layers.BaseLayer)
  :special-members: __init__
  :undoc-members: __init__

.. autoclass:: molgraph.layers.GMMConv(molgraph.layers.BaseLayer)
  :special-members: __init__
  :undoc-members: __init__

.. autoclass:: molgraph.layers.GraphTransformerConv(molgraph.layers.BaseLayer)
  :special-members: __init__
  :undoc-members: __init__


Message-passing
~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: molgraph.layers.MPNNConv(molgraph.layers.BaseLayer)
  :special-members: __init__
  :undoc-members: __init__


Geometric
~~~~~~~~~~~~~
.. autoclass:: molgraph.layers.DTNNConv(molgraph.layers.BaseLayer)
  :special-members: __init__
  :undoc-members: __init__

.. autoclass:: molgraph.layers.GCFConv(molgraph.layers.BaseLayer)
  :special-members: __init__
  :undoc-members: __init__


Readout
~~~~~~~~~~~~
.. autoclass:: molgraph.layers.SegmentPoolingReadout(tensorflow.keras.layers.Layer)
  :special-members: __init__
  :undoc-members: __init__

.. autoclass:: molgraph.layers.TransformerEncoderReadout(tensorflow.keras.layers.Layer)
  :special-members: __init__
  :undoc-members: __init__

.. autoclass:: molgraph.layers.SetGatherReadout(tensorflow.keras.layers.Layer)
  :special-members: __init__
  :undoc-members: __init__
