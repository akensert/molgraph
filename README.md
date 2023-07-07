# MolGraph: Graph Neural Networks for Molecular Machine Learning

*This is an early release; things are still being updated, added and experimented with. Hence, API compatibility may break in the future.*

*Any feedback is welcomed!*

## Manuscript
See [pre-print](https://arxiv.org/abs/2208.09944)

## Documentation
See [readthedocs](https://molgraph.readthedocs.io/en/latest/)

## Implementations

- **Tensor**
    - **GraphTensor**
        - A composite tensor holding graph data.
        - Has a ragged (multiple graphs) and a non-ragged state (single disjoint graph)
        - Can conveniently go between both states (merge() and separate())
        - Can propagate node information (features) based on edges (propagate())
        - Can add, update and remove graph data (update(), remove())
        - Has an associated GraphTensorSpec which it compatible with Keras and TensorFlow API.
            - This includes keras.Sequential, keras.Functional, tf.data.Dataset, and tf.saved_model API.
- **Layers**
    
    - **Convolutional**
        - GCNConv ([GCNConv](http://github.com/akensert/molgraph/tree/main/molgraph/layers/convolutional/gcn_conv.py))
        - GINConv ([GINConv](https://github.com/akensert/molgraph/tree/main/molgraph/layers/convolutional/gin_conv.py))
        - GCNIIConv ([GCNIIConv](https://github.com/akensert/molgraph/tree/main/molgraph/layers/convolutional/gcnii_conv.py))
        - GraphSageConv ([GraphSageConv](https://github.com/akensert/molgraph/tree/main/molgraph/layers/convolutional/graph_sage_conv.py))
    - **Attentional**
        - GATConv ([GATConv](https://github.com/akensert/molgraph/tree/main/molgraph/layers/attentional/gat_conv.py))
        - GATv2Conv ([GATv2Conv](https://github.com/akensert/molgraph/tree/main/molgraph/layers/attentional/gatv2_conv.py))
        - GTConv ([GTConv](https://github.com/akensert/molgraph/tree/main/molgraph/layers/attentional/gt_conv.py))
        - GMMConv ([GMMConv](https://github.com/akensert/molgraph/tree/main/molgraph/layers/attentional/gmm_conv.py))
        - GatedGCNConv ([GatedGCNConv](https://github.com/akensert/molgraph/tree/main/molgraph/layers/attentional/gated_gcn_conv.py))
        - AttentiveFPConv ([AttentiveFPConv](https://github.com/akensert/molgraph/tree/main/molgraph/layers/attentional/attentive_fp_conv.py))
    - **Message-passing**
        - MPNNConv ([MPNNConv](https://github.com/akensert/molgraph/tree/main/molgraph/layers/message_passing/mpnn_conv.py))
        - EdgeConv ([EdgeConv](https://github.com/akensert/molgraph/tree/main/molgraph/layers/message_passing/edge_conv.py))
    - **Distance-geometric**
        - DTNNConv ([DTNNConv](https://github.com/akensert/molgraph/tree/main/molgraph/layers/geometric/dtnn_conv.py))
        - GCFConv ([GCFConv](https://github.com/akensert/molgraph/tree/main/molgraph/layers/geometric/gcf_conv.py))
    - **Pre- and post-processing**
        - In addition to the aforementioned GNN layers, there are also several other layers which improves model-building. See `readout/`, `preprocessing/`, `postprocessing/`, `positional_encoding/`.
- **Models**
    - Although model building is easy with MolGraph, there are some built-in GNN models:
        - **DGIN**
        - **DMPNN**
        - **MPNN**
    - And models for improved interpretability of GNNs:
        - **SaliencyMapping**
        - **IntegratedSaliencyMapping**
        - **SmoothGradSaliencyMapping**
        - **GradientActivationMapping** (Recommended)

## Changelog
For a detailed list of changes, see the [CHANGELOG.md](https://github.com/akensert/molgraph/blob/main/CHANGELOG.md).

**Important notes**
- Since version **0.4.2**, default normalization for the GNN layers is layer normalization. This significantly improved the performance on some of the MoleculeNet datasets.

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
scores = gnn_model.evaluate(x_test, y_test)

# Compute gradient activation maps
gam_model = models.GradientActivationMapping(
    model=gnn_model, layer_names=['gat_conv_1', 'gat_conv_2'])

maps = gam_model(x_train)
```

## Requirements/dependencies
- **Python** (version >= 3.6 recommended)
- **TensorFlow** (version >= 2.7.0 recommended)
- **RDKit** (version >= 2022.3.3 recommended)
- **NumPy** (version >= 1.21.2 recommended)
- **Pandas** (version >= 1.0.3 recommended)

## Tested with
- **Ubuntu 20.04 - Python 3.8.10**
- **MacOS Monterey (12.3.1) - Python 3.10.3**
