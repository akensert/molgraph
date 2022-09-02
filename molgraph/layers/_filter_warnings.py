from warnings import filterwarnings

filterwarnings('ignore',
    message='Converting sparse IndexedSlices.*' +
            'to a dense Tensor of unknown shape. ' +
            'This may consume a large amount of memory.')

import logging
#logging.getLogger('tensorflow').setLevel(logging.ERROR)

def ignore_gradient_warning(record):
    # If e.g. `update_edge_features` is passed to the last GAT layer, edge
    # features will be updated via a kernel without having an effect on the
    # output. Hence, gradients will not exist for that kernel. To avoid this
    # kernel, pass 'update_edge_features=False` to the last GAT layer, or any
    # other layer which updates each features.
    return not record.msg.startswith('Gradients do not exist for variables')

logging.getLogger('tensorflow').addFilter(ignore_gradient_warning)
