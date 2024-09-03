import tensorflow as tf
from tensorflow import keras

from molgraph.internal import register_keras_serializable 

from molgraph.tensors.graph_tensor import GraphTensor


@register_keras_serializable(package='molgraph')
class SuperNodeReadout(keras.layers.Layer):

    '''Extracts a set of ("super") node features for each subgraph.
    
    Extracted "super" node features can then be used as input for a 
    sequence model such as an RNN.

    For instance, for each peptide, node features corresponding to 
    "residue nodes" can be extracted. The "residue nodes" would correspond to 
    the "super nodes".
    
    The inputted graph could looks something like this:

             peptide1  ...
              |      \    
          residue1   residue2  ...
         /   |   \        |  \ 
    atom1__atom2__atom3__  ...

    And the resulting output will be of shape: 
    
    (n_peptides, n_residues, n_features)
    
    '''

    def __init__(self, indicator_field: str, **kwargs):
        super().__init__(**kwargs)
        self.indicator_field = indicator_field
 
    def call(self, tensor: GraphTensor) -> tf.Tensor:
        if isinstance(tensor.node_feature, tf.Tensor):
            tensor = tensor.separate()
        mask = tf.cast(getattr(tensor, self.indicator_field), tf.bool)
        return tf.ragged.boolean_mask(tensor.node_feature, mask).to_tensor()
    
    def get_config(self):
        config = super().get_config()
        config.update({'indicator_field': self.indicator_field})
        return config
    