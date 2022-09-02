# MolGraph: Graph Neural Networks for Molecular Machine Learning

*This is an early release; things are still being updated and added. Hence, API compatibility may break in the future.*

*Any feedback is welcomed!*

## Manuscript
See [pre-print](https://arxiv.org/abs/2208.09944)

## Documentation
See [readthedocs](https://molgraph.readthedocs.io/en/latest/)

## Installation

Install via **pip**:

<pre>
pip install molgraph
</pre>

Install via **docker**:

<pre>
git clone https://github.com/akensert/molgraph.git
cd molgraph/docker
docker build -t molgraph-tf[-gpu][-jupyter]/molgraph:0.0 molgraph-tf[-gpu][-jupyter]/
docker run -it <b>[-p 8888:8888]</b> molgraph-tf[-gpu]<b>[-jupyter]</b>/molgraph:0.0
</pre>

Now run your first program with **MolGraph**:

```python
from tensorflow import keras
from molgraph import chemistry
from molgraph import layers
from molgraph import models

# Obtain dataset, specifically ESOL
qm7 = chemistry.datasets.get('esol')

# Define molecular graph encoder
atom_encoder = chemistry.AtomicFeaturizer([
    chemistry.features.Symbol(),
    chemistry.features.Hybridization(),
    # ...
])

bond_encoder = chemistry.AtomicFeaturizer([
    chemistry.features.BondType(),
    # ...
])

encoder = chemistry.MolecularGraphEncoder(atom_encoder, bond_encoder)

# Obtain features and associated labels
x_train = encoder(qm7['train']['x'])
y_train = qm7['train']['y']

x_test = encoder(qm7['test']['x'])
y_test = qm7['test']['y']

# Build model via Keras API
gnn_model = keras.Sequential([
    keras.layers.Input(type_spec=x_train.spec),
    layers.GATConv(name='gat_conv_1'),
    layers.GATConv(name='gat_conv_2'),
    layers.Readout(),
    keras.layers.Dense(units=1024, activation='relu'),
    keras.layers.Dense(units=y_train.shape[-1])
])

# Compile, fit and evaluate
gnn_model.compile(optimizer='adam', loss='mae')
gnn_model.fit(x_train, y_train, epochs=50)
gnn_model.evaluate(x_test, y_test)

# Compute gradient activation maps
gam_model = models.GradientActivationMapping(
    model=gnn_model, layer_names=['gat_conv_1', 'gat_conv_2'])

maps = gam_model.predict(x_train)
```

## Requirements/dependencies
- **Python** (version ~= 3.8.10)
- **TensorFlow** (version ~= 2.7.0)
- **RDKit** (version ~= 2022.3.3)
- **NumPy** (version ~= 1.21.2)
- **Pandas** (version ~= 1.0.3)

## Tested with
- **Ubuntu 20.04 - Python 3.8.10**
- **MacOS Monterey (12.3.1) - Python 3.10.3**
