<img src="https://github.com/akensert/molgraph/blob/main/docs/source/_static/molgraph-logo-pixel.png" alt="molgraph-title" width="90%">

**Graph Neural Networks** with **TensorFlow** and **Keras**. Focused on **Molecular Machine Learning**.

> Currently, Keras 3 does not support extension types. As soon as it does, it is hoped that MolGraph will migrate to Keras 3.

## Highlights

Build a Graph Neural Network with Keras' [Sequential](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential) API:

```python
from molgraph import GraphTensor
from molgraph import layers
from tensorflow import keras

g = GraphTensor(node_feature=[[4.], [2.]], edge_src=[0], edge_dst=[1])

model = keras.Sequential([
    layers.GNNInput(type_spec=g.spec),
    layers.GATv2Conv(units=32),
    layers.GATv2Conv(units=32),
    layers.Readout(),
    keras.layers.Dense(units=1),
])

pred = model(g)

# Save and load Keras model
model.save('/tmp/gatv2_model.keras')
loaded_model = keras.models.load_model('/tmp/gatv2_model.keras')
loaded_pred = loaded_model(g)
assert pred == loaded_pred
```

Combine outputs of GNN layers to improve predictive performance:

```python
model = keras.Sequential([
    layers.GNNInput(type_spec=g.spec),
    layers.GNN([
        layers.FeatureProjection(units=32),
        layers.GINConv(units=32),
        layers.GINConv(units=32),
        layers.GINConv(units=32),
    ]),
    layers.Readout(),
    keras.layers.Dense(units=128),
    keras.layers.Dense(units=1),
])

model.summary()
```

## Paper
See [arXiv](https://arxiv.org/abs/2208.09944)

## Documentation
See [readthedocs](https://molgraph.readthedocs.io/en/latest/)

## Overview 

<img src="https://github.com/akensert/molgraph/blob/main/docs/source/_static/molgraph-overview.png" alt="molgraph-overview" width="90%">

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
        - **GradientActivationMapping**


## Requirements

- **Python** (version >= 3.10)
    - **TensorFlow** (version 2.15.*)
    - **RDKit** (version 2023.9.*)
    - **Pandas**
    - **IPython**


## Installation

For **CPU** users:
<pre>
pip install molgraph
</pre>

For **GPU** users:
<pre>
pip install molgraph[gpu]
</pre>

