from warnings import filterwarnings

filterwarnings('ignore',
    message='Converting sparse IndexedSlices.*' +
            'to a dense Tensor of unknown shape. ' +
            'This may consume a large amount of memory.')
