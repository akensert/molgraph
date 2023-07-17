import logging
import absl.logging
from warnings import filterwarnings
from rdkit import RDLogger
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

filterwarnings(
    'ignore',
    message=(
        'Encoding a StructuredValue with type '
        'molgraph.tensors.graph_tensor.GraphTensorSpec; '
        'loading this StructuredValue will require that this '
        'type be imported and registered.*'
    )
)

filterwarnings('ignore',
    message='Converting sparse IndexedSlices.*' +
            'to a dense Tensor of unknown shape. ' +
            'This may consume a large amount of memory.')

absl.logging.set_verbosity(absl.logging.ERROR)

#logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(message)s')

def ignore_gradient_warning(record):
    # If e.g. `update_edge_features` is passed to the last GAT layer, edge
    # features will be updated via a kernel without having an effect on the
    # output. Hence, gradients will not exist for that kernel. To avoid this
    # kernel, pass 'update_edge_features=False` to the last GAT layer, or any
    # other layer which updates each features.
    return not record.msg.startswith('Gradients do not exist for variables')

logging.getLogger('tensorflow').addFilter(ignore_gradient_warning)

RDLogger.DisableLog("rdApp.*")


