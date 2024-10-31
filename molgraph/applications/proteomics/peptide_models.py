import tensorflow as tf 
from tensorflow import keras

from molgraph import layers
from molgraph.applications.proteomics.peptide_layers import _ResidueReadout
from molgraph.applications.proteomics.definitions import _residue_node_mask


_default_model_config = {
    "gnn": {
        "num_layers": 2,
        "layer_type": layers.GATv2Conv,
        "layer_kwargs": {
            "units": 128,
            "normalization": None,
            "self_projection": True,
            "kernel_regularizer": keras.regularizers.L2(1e-5)
        },
        "trainable": True,
    },
    "rnn": {
        "num_layers": 1,
        "layer_type": keras.layers.LSTM,
        "layer_kwargs": {
            "units": 128,
        }
    },
    "dnn": {
        "num_layers": 2,
        "layer_type": keras.layers.Dense,
        "layer_kwargs": {
            "units": 1024,
            "activation": "relu",
            "kernel_regularizer": keras.regularizers.L2(1e-5),
        },
        "output_units": 1,
        "output_activation": "linear",
    }
}


def PeptideModel(config: dict = None, **kwargs) -> keras.Model:
    
    spec = kwargs.pop("spec", None)
    if spec is None:
        raise ValueError("A `GraphTensor.Spec` needs to be passed as `spec`.")

    has_super_nodes = _residue_node_mask in spec.auxiliary

    if not config:
        config = _default_model_config

    config["rnn"]["layer_kwargs"].pop("return_sequences", None)

    message_layer = config["gnn"]["layer_type"]
    if isinstance(message_layer, str):
        message_layer = getattr(layers, message_layer.rstrip('Conv') + 'Conv')
        
    graph_layers = [
        layers.NodeFeatureProjection(
            units=config["gnn"]["layer_kwargs"].get("units"),
            kernel_regularizer=config["gnn"]["layer_kwargs"].get("kernel_regularizer"),
        )
    ]
    graph_layers += [
        message_layer(**config["gnn"]["layer_kwargs"])
        for _ in range(config["gnn"]["num_layers"])
    ]
    for layer in graph_layers[1:]:
        layer.trainable = config["gnn"]["trainable"] 

    if has_super_nodes:
        readout = layers.SuperNodeReadout(_residue_node_mask)
    else:
        readout = _ResidueReadout()
        
    rnn_layer = config["rnn"]["layer_type"]
    if isinstance(rnn_layer, str):
        rnn_layer = getattr(keras.layers, rnn_layer)

    rnn_layers = [
        keras.layers.Bidirectional(
            rnn_layer(**config["rnn"]["layer_kwargs"], return_sequences=True)
        )
        for _ in range(config["rnn"]["num_layers"])
    ]
    rnn_layers += [keras.layers.GlobalAveragePooling1D()]

    dnn_layer = config["dnn"]["layer_type"]
    if isinstance(dnn_layer, str):
        dnn_layer = getattr(keras.layers, dnn_layer)

    dense_layers = [
        dnn_layer(**config["dnn"]["layer_kwargs"])
        for _ in range(config["dnn"]["num_layers"] - 1)
    ]

    config["dnn"]["layer_kwargs"]["units"] = config["dnn"]["output_units"]
    config["dnn"]["layer_kwargs"]["activation"] = config["dnn"]["output_activation"]

    dense_layers += [
        dnn_layer(**config["dnn"]["layer_kwargs"])
    ]
    return tf.keras.Sequential([
        layers.GNNInputLayer(type_spec=spec),
        layers.GNN(graph_layers),
        readout,
        *rnn_layers,
        *dense_layers,
    ])

