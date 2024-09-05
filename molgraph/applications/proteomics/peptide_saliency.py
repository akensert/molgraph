import tensorflow as tf
import keras

from molgraph.tensors.graph_tensor import GraphTensor 
from molgraph.applications.proteomics.definitions import _residue_node_indicator 



class PeptideSaliency:
    
    def __init__(
        self, 
        model: keras.Sequential, 
        output_activation: str | None = None, 
        absolute: bool = False,
        reduce: bool = True,
        target: str = 'node_feature',
        call_eagerly: bool = True
    ) -> None:
        # If object will be called repeatedly, set `call_eagerly` to False for faster execution
        self.model = model
        self.activation = keras.activations.get(output_activation)
        self.absolute = absolute
        self.reduce = reduce
        self.target = target
        self.call_eagerly = call_eagerly
        self.saliency_field = (
            'node_saliency' if 'node' in self.target else 'edge_saliency')
        self.feature_field = (
            'node_feature' if 'node' in self.target else 'edge_feature')
        if not self.call_eagerly:
            self._call = tf.function(self._call)

    def __call__(
        self, 
        x: GraphTensor, 
        y: tf.Tensor | None = None, 
    ) -> tf.RaggedTensor:
        return self._call(x, y)
    
    def _call(
        self, 
        x: GraphTensor, 
        y: tf.Tensor | None = None, 
    ) -> tf.RaggedTensor:
        if isinstance(x.node_feature, tf.RaggedTensor):
            x = x.merge()
        x = x.update({self.saliency_field: self._compute_saliency(x, y)})
        return getattr(self._filter(x).separate(), self.saliency_field)
        
    def _compute_saliency(
        self, 
        x: GraphTensor, 
        y: tf.Tensor | None = None,
        target: str = 'node_feature',
    ) -> tf.Tensor:
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(getattr(x, self.feature_field))
            y_pred = self.model(x)
            y_pred = self.activation(y_pred)
            if y is not None and len(y.shape) >= 2:
                y_pred = tf.gather_nd(y_pred, tf.where(y == 1))
        gradients = tape.gradient(y_pred, getattr(x, self.feature_field))
        if self.absolute:
            gradients = tf.abs(gradients)
        if self.reduce:
            gradients = tf.reduce_sum(gradients, axis=-1)
        return gradients 
    
    def _filter(self, x: GraphTensor) -> GraphTensor:
        mask = tf.cast(getattr(x, _residue_node_indicator), tf.bool)
        node_level = (
            True if ('node' in self.target or 'edge' in self.target) else False)
        mask = (mask == False if node_level else mask)
        return tf.boolean_mask(x, mask, axis='node')

