# Graph Neural Networks (for small molecular graphs) with TensorFlow and Keras

## In progress/unfinished
This is an early release. API compatibility may break in the future.

## Examples
see `docs/source/examples`

## Installation

### Requirements
- **Python** (version ~= 3.8.10)
- **TensorFlow** (version ~= 2.8.0)
- **RDKit** (rdkit-pypi) (version ~= 2021.3.4)
- **NumPy** (version ~= 1.22.2)

### Tested on
- **Ubuntu 20.04**

### Install package
From terminal run `pip install git+https://github.com/akensert/molgraph.git`


## Minimalistic implementation
A complete GNN implementation for small molecular graphs in about 20 lines of code:

```python
import molgraph

qm7 = molgraph.chemistry.datasets.get('qm7')

# Define molecular graph encoder
atom_encoder = molgraph.chemistry.AtomFeaturizer([
    molgraph.chemistry.features.Symbol(),
    molgraph.chemistry.features.Hybridization(),
    # ...
])

bond_encoder = molgraph.chemistry.BondFeaturizer([
    molgraph.chemistry.features.BondType(),
    # ...
])

encoder = molgraph.chemistry.MolecularGraphEncoder(atom_encoder, bond_encoder)

x_train = encoder(qm7['train']['x'])
y_train = qm7['train']['y']

x_test = encoder(qm7['test']['x'])
y_test = qm7['test']['y']

# Keras API
gnn_model = tf.keras.Sequential([
  keras.layers.Input(type_spec=x_train.spec),
  molgraph.layers.GATConv(),
  molgraph.layers.GATConv(),
  molgraph.layers.Readout(),
  keras.layers.Dense(units=1024, activation='relu'),
  keras.layers.Dense(units=y_train.shape[-1])
])

gnn_model.compile(optimizer='adam', loss='mae')
gnn_model.fit(x_train, num_epochs=50)
gnn_model.evaluate(x_test)
```

## Minimalistic tutorial

Datasets:

```python
from molgraph.chemistry import datasets

# datasets is a factory
print(datasets.registered_datasets)

qm7 = datasets.get('qm7')
```

Atomic features:
```python
from molgraph.chemistry import features
from molgraph.chemistry import molecule_from_string

rdkit_mol = molecule_from_string('CCO')
atom = rdkit_mol.GetAtoms()[0]
bond = rdkit_mol.GetBonds()[0]

single_atom_feature = features.Symbol()
print(single_atom_feature(atom))
single_bond_feature = features.BondType()
print(single_bond_feature(bond))

# features has two factories
print(features.atom_features)
print(features.bond_features)

# get all registered [atom] features
all_atom_features = features.atom_features.unpack()

# get a single atom features
single_atom_feature_2 = features.atom_features.get('symbol')

```

Atomic featurizer
```python
from molgraph.chemistry import AtomFeaturizer

atom_featurizer = AtomFeaturizer([
    features.Symbol(),
    features.Hybridization(),
])

# Get encoding of atom
print(atom_featurizer(atom))
# Get encoding of all atoms
print(atom_featurizer.encode_atoms(rdkit_mol.GetAtoms()))
```

Molecular [graph] encoder:
```python
from molgraph.chemistry import MolecularGraphEncoder
from molgraph.chemistry import MolecularGraphEncoder3D

encoder = MolecularGraphEncoder(atom_encoder)

# pass list of molecules (in this case SDF strings) to encoder
print(encoder(qm7['train']['x']))
# pass single molecule (in this case RDKit molecule object) to encoder
print(encoder(rdkit_mol))

encoder_3d = MolecularGraphEncoder3D(atom_encoder)
# Requires molecule to have conformer
try:
  print(encoder(rdkit_mol))
except:
  pass

# pass conformer generator if needed
from molgraph.chemistry import ConformerGenerator

encoder_3d = MolecularGraphEncoder3D(
  atom_encoder, conformer_generator=ConformerGenerator())

print(encoder(rdkit_mol))
```

GNN layer:
```python
from molgraph import layers

gat_layer = layers.GATConv(units=128)

# obtain molecular graph (GraphTensor)
x_train = encoder(qm7['train']['x'])

# pass GraphTensor with nested ragged tensors (default)
gat_layer(x_train)

# pass GraphTensor with nested tensors
# merge all subgraphs (molecules) into a single disjoint graph
x_train = x_train.merge()
get_layer(x_train)
```

GNN model and gradient activation maps:
```python
from tensorflow import keras

# divide disjoint graph into subgraphs again (nested ragged tensors)
# this allows for batching the GraphTensor
x_train = x_train.unmerge()

gnn_model = keras.Sequential([
    keras.layers.Input(type_spec=x_train.unspecific_spec)
    keras.layers.GATConv(name='gat_conv_1'),
    keras.layers.GATConv(name='gat_conv_2'),
    keras.layers.Readout(),
    keras.layers.Dense(qm7['train']['y'].shape[-1]),
])

gnn_model(x_train)
gnn_model.compile('adam', 'mse')
gnn_model.fit(x_train, qm7['train']['y'], epochs=100)

# gradient activation maps
from molgraph import interpretability

gam_model = interpretability.GradientActivationMaps(
    model=gnn_model, layer_names=['gat_conv_1', 'gat_conv_2']
)

maps = gam_model.predict(x_train)
```
