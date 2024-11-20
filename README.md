<img src="https://github.com/akensert/molgraph/blob/main/docs/source/_static/molgraph-logo-pixel.png" alt="molgraph-title" width="90%">

**Graph Neural Networks** with **TensorFlow** and **Keras**. Focused on **Molecular Machine Learning**.

## Quick start
Benchmark the performance of MolGraph [here](https://github.com/akensert/molgraph/blob/main/examples/GNN-Benchmarking.ipynb), and implement a complete model pipeline with MolGraph [here](https://github.com/akensert/molgraph/blob/main/examples/QSAR-GNN-Tutorial.ipynb). 


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

## Installation
For **CPU** users:
<pre>
pip install molgraph
</pre>
For **GPU** users:
<pre>
pip install molgraph[gpu]
</pre>

## Implementations
- **Tensors**
    - [Graph tensor](http://github.com/akensert/molgraph/tree/main/molgraph/tensors/graph_tensor.py)
        - A composite tensor holding graph data; compatible with `tf.data.Dataset`, `keras.Sequential` and much more.
- **Layers**
    - [Graph convolutional layers](http://github.com/akensert/molgraph/tree/main/molgraph/layers/convolutional/)
    - [Graph attentional layers](http://github.com/akensert/molgraph/tree/main/molgraph/layers/attentional/)
    - [Graph message passing layers](http://github.com/akensert/molgraph/tree/main/molgraph/layers/message_passing/)
    - [Graph readout layers](http://github.com/akensert/molgraph/tree/main/molgraph/layers/readout/)
    - [Preprocessing layers](https://github.com/akensert/molgraph/tree/main/molgraph/layers/preprocessing/)
    - [Postprocessing layers](https://github.com/akensert/molgraph/tree/main/molgraph/layers/postprocessing/)
    - [Positional encoding layers](https://github.com/akensert/molgraph/tree/main/molgraph/layers/positional_encoding)
- **Models**
    - [Graph neural networks](https://github.com/akensert/molgraph/tree/main/molgraph/models/)
    - [Saliency mapping](https://github.com/akensert/molgraph/tree/main/molgraph/models/interpretability/)

## Overview 
<img src="https://github.com/akensert/molgraph/blob/main/docs/source/_static/molgraph-overview.png" alt="molgraph-overview" width="90%">

## Documentation
See [readthedocs](https://molgraph.readthedocs.io/en/latest/)

## Papers
- [MolGraph: a Python package for the implementation of molecular graphs and graph neural networks with TensorFlow and Keras](https://doi.org/10.48550/arXiv.2208.09944)
- [A hands-on tutorial on quantitative structure-activity relationships using fully expressive graph neural networks](https://doi.org/10.1016/j.aca.2024.343046)
