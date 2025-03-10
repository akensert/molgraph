__version__ = '0.8.1'

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

from molgraph import layers
from molgraph import losses
from molgraph import metrics
from molgraph import models
from molgraph import chemistry

from molgraph.tensors import GraphTensor
