###################
Installation
###################

Install via **pip**:

.. code-block::

  pip install molgraph

Or via **docker**:

.. code-block::

  git clone https://github.com/akensert/molgraph.git
  cd molgraph/docker
  docker build -t molgraph-tf[-gpu][-jupyter]/molgraph:0.0 molgraph-tf[-gpu][-jupyter]/
  docker run -it [-p 8888:8888] molgraph-tf[-gpu][-jupyter]/molgraph:0.0


Now run your first program with **MolGraph**:

.. code-block::

    from tensorflow import keras
    from molgraph import chemistry
    from molgraph import layers
    from molgraph import models

    # Obtain dataset, specifically ESOL
    esol = chemistry.datasets.get('esol')

    # Define molecular graph encoder
    atom_encoder = chemistry.Featurizer([
        chemistry.features.Symbol(),
        chemistry.features.Hybridization(),
        # ...
    ])

    bond_encoder = chemistry.Featurizer([
        chemistry.features.BondType(),
        # ...
    ])

    encoder = chemistry.MolecularGraphEncoder(atom_encoder, bond_encoder)

    # Obtain features and associated labels
    x_train = encoder(esol['train']['x'])
    y_train = esol['train']['y']

    x_test = encoder(esol['test']['x'])
    y_test = esol['test']['y']

    # Build model via Keras API
    gnn_model = keras.Sequential([
        layers.GATConv(units=32, name='gat_conv_1'),
        layers.GATConv(units=32, name='gat_conv_2'),
        layers.Readout(),
        keras.layers.Dense(units=1024, activation='relu'),
        keras.layers.Dense(units=y_train.shape[-1])
    ])

    # Compile, fit and evaluate
    gnn_model.compile(optimizer='adam', loss='mae')
    gnn_model.fit(x_train, y_train, epochs=50)
    scores = gnn_model.evaluate(x_test, y_test)

    # Compute gradient activation maps
    gam_model = models.GradientActivationMapping(
        model=gnn_model, layer_names=['gat_conv_1', 'gat_conv_2'])

    maps = gam_model(x_train.separate())
