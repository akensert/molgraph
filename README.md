# MolGraph

**Graph Neural Networks** with **TensorFlow** and **Keras**. Focused on **Molecular Machine Learning**.

<img src="https://github.com/akensert/molgraph/blob/main/media/molgraph.jpg" alt="molgraph" width="820">

## Highlights

Build a Graph Neural Network with Keras' [Sequential](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential) API:

```python
from molgraph import GraphTensor
from molgraph import layers
from tensorflow import keras

model = keras.Sequential([
    layers.GINConv(units=32),
    layers.GINConv(units=32),
    layers.Readout(),
    keras.layers.Dense(units=1),
])
output = model(
    GraphTensor(node_feature=[[4.], [2.]], edge_src=[0], edge_dst=[1])
)
```

## Paper
See [arXiv](https://arxiv.org/abs/2208.09944)

## Documentation
See [readthedocs](https://molgraph.readthedocs.io/en/latest/)

## Implementations

- **Graph tensor** ([GraphTensor](http://github.com/akensert/molgraph/tree/main/molgraph/tensors/graph_tensor.py))
    - A composite tensor holding graph data.
    - Has a ragged state (multiple graphs) and a non-ragged state (single disjoint graph).
    - Can conveniently go between both states (merge(), separate()).
    - Can propagate node states (features) based on edges (propagate()).
    - Can add, update and remove graph data (update(), remove()).
    - Compatible with TensorFlow's APIs (including Keras). For instance, graph data (encoded as a GraphTensor) can now seamlessly be used with keras.Sequential, keras.Functional, tf.data.Dataset, and tf.saved_model APIs.
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
        - In addition to the aforementioned GNN layers, there are also several other layers which improves model-building. See [readout/](https://github.com/akensert/molgraph/tree/main/molgraph/layers/readout), [preprocessing/](https://github.com/akensert/molgraph/tree/main/molgraph/layers/preprocessing), [postprocessing/](https://github.com/akensert/molgraph/tree/main/molgraph/layers/postprocessing), [positional_encoding/](https://github.com/akensert/molgraph/tree/main/molgraph/layers/positional_encoding).
- **Models**
    - Although model building is easy with MolGraph, there are some built-in GNN [models](https://github.com/akensert/molgraph/tree/main/molgraph/models):
        - **GIN**
        - **MPNN**
        - **DMPNN**
    - And models for improved interpretability of GNNs:
        - **SaliencyMapping**
        - **IntegratedSaliencyMapping**
        - **SmoothGradSaliencyMapping**
        - **GradientActivationMapping** (Recommended)

## Requirements/dependencies
- **Python** (version ~= 3.10.0)
    - **TensorFlow** (version ~= 2.15.0)
    - **RDKit** (version ~= 2022.3.5)
    - **Pandas** (version ~= 1.0.3)
    - **IPython** (version ~= 8.12.0)

> MolGraph should work with the more recent TensorFlow and RDKit versions. If not, try installing earlier versions of TensorFlow and RDKit.

## Installation

For **GPU** users:
<pre>
pip install molgraph[gpu]
</pre>

For **CPU** users:
<pre>
pip install molgraph
</pre>

Now run your first program with **MolGraph**:

```python
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

# Obtain graphs and associated labels
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
```
