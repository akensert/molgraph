from warnings import filterwarnings

filterwarnings('ignore',
    message='Converting sparse IndexedSlices.*' +
            'to a dense Tensor of unknown shape. ' +
            'This may consume a large amount of memory.')

import logging
#logging.getLogger('tensorflow').setLevel(logging.ERROR)

def ignore_gradient_warning(record):
    return not record.msg.startswith('Gradients do not exist for variables')

logging.getLogger('tensorflow').addFilter(ignore_gradient_warning)
