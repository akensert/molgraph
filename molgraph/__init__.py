__version__ = '0.8.3'

import sys

tf_module = sys.modules.get('tensorflow', None)
if tf_module is not None:
    get_version_fn = getattr(tf_module.keras, 'version', None)
    if get_version_fn is not None:
        version = get_version_fn()
        if int(version[0]) >= 3: 
            raise ImportError(
                "MolGraph currently requires Keras 2. For TensorFlow>2.15, "
                "make sure to set TF_USE_LEGACY_KERAS=1 before importing TensorFlow:\n\n"
                "\timport os\n"
                "\tos.environ['TF_USE_LEGACY_KERAS'] = '1'\n"
                "\timport tensorflow as tf\n"
                "\timport molgraph\n"
                "\t...\n\n"
                "Alternatively, import molgraph before tensorflow:\n\n"
                "\timport molgraph\n"
                "\timport tensorflow as tf\n"
                "\t..."
            )

import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

from molgraph import _filter_warnings
del _filter_warnings

import molgraph.layers
import molgraph.tensors
import molgraph.losses
import molgraph.metrics
import molgraph.models
import molgraph.chemistry

from molgraph.tensors.graph_tensor import GraphTensor
