from molgraph.tensors.graph_tensor import GraphTensor
from molgraph.layers.gnn_input_layer import GNNInputLayer


def GNNInput(type_spec: GraphTensor.Spec):
    return GNNInputLayer(type_spec=type_spec).output 
