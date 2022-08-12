Models
================

Interpretability
~~~~~~~~~~~~~~~~~~~~~~

**Code snippet:**

.. code-block::

  from molgraph import models
  from molgraph import layers
  from molgraph import chemistry

  encoder = chemistry.MolecularGraphEncoder(
    atom_encoder=chemistry.AtomFeaturizer(
        chemistry.atom_features.unpack()
    )
  )

  esol = chemistry.datasets.get('esol')

  esol['train']['x'] = encoder(esol['train']['x'])
  esol['test']['x'] = encoder(esol['test']['x'])

  # Pass GraphTensor to model
  gnn_model = tf.keras.Sequential([
    tf.keras.layers.Input(type_spec=esol['train']['x'].spec),
    layers.GCNConv(units=128, name='gcn_conv_1'),
    layers.GCNConv(units=128, name='gcn_conv_2'),
    layers.GCNConv(units=128, name='gcn_conv_3'),
    layers.Readout('mean'),
    tf.keras.layers.Dense(units=512),
    tf.keras.layers.Dense(units=1)
  ])
  gnn_model.compile(optimizer='adam', loss='mse')
  gnn_model.fit(esol['train']['x'], esol['train']['y'], epochs=10)

  gam_model = models.GradientActivationMapping(
    model=gnn_model,
    layer_names=['gcn_conv_1', 'gcn_conv_2', 'gcn_conv_3'],
    output_activation='sigmoid',
    discard_negative_values=False
  )
  # Interpretability models can only be predicted with
  maps = gam_model.predict(esol['test']['x'])


.. autoclass:: molgraph.models.GradientActivationMapping(tensorflow.keras.Model)
  :members: predict
  :special-members: __init__
  :undoc-members: __init__


.. autoclass:: molgraph.models.SaliencyMapping(tensorflow.keras.Model)
  :members: predict
  :special-members: __init__
  :undoc-members: __init__


.. autoclass:: molgraph.models.IntegratedSaliencyMapping(tensorflow.keras.Model)
  :members: predict
  :special-members: __init__
  :undoc-members: __init__


.. autoclass:: molgraph.models.SmoothGradSaliencyMapping(tensorflow.keras.Model)
  :members: predict
  :special-members: __init__
  :undoc-members: __init__
