import tensorflow as tf
import keras

from molgraph.tensors.graph_tensor import GraphTensor 
from molgraph.applications.proteomics.definitions import _residue_node_indicator 


class PeptideSaliency:
    
    '''Peptide saliency based on gradient (class) activation maps (Grad-CAM).
    ''' 
    
    def __init__(
        self, 
        model: keras.Sequential, 
        output_activation: str | None = None, 
        discard_negative_values: bool = False,
        reduce: bool = True,
        node_level: bool = True,
        call_eagerly: bool = True
    ) -> None:
        # If object will be called repeatedly, set `call_eagerly` to False for faster execution
        self.model = model
        self.activation = keras.activations.get(output_activation)
        self.discard_negative_values = discard_negative_values
        self.reduce = reduce
        self.node_level = node_level
        self.call_eagerly = call_eagerly
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
        x = x.update({'node_saliency': self._saliency_maps(x, y)})
        return getattr(self._filter(x).separate(), 'node_saliency')
        
    def _saliency_maps(
        self, 
        x: GraphTensor, 
        y: tf.Tensor | None = None,
    ) -> tf.Tensor:
        graph_indicator = None
        features = []
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            for i, layer in enumerate(self.model.layers):
                if not isinstance(x, GraphTensor):
                    x = layer(x)
                    continue 
                if graph_indicator is None:
                    graph_indicator = x.graph_indicator
                if hasattr(layer, '_taped_call'):
                    x, taped_features = layer._taped_call(x, tape)
                    features.extend(taped_features)
                else:
                    tape.watch(x.node_feature)
                    features.append(x.node_feature)
                    x = layer(x)
            y_pred = self.activation(x)
            if y is not None and len(y.shape) >= 2:
                y_pred = tf.gather_nd(y_pred, tf.where(y == 1))
        gradients = tape.gradient(y_pred, features)
        features = tf.concat(features, axis=-1)
        gradients = tf.concat(gradients, axis=-1)
        alpha = tf.math.segment_mean(gradients, graph_indicator)
        alpha = tf.gather(alpha, graph_indicator)
        maps = tf.where(gradients != 0, alpha * features, gradients)
        if self.reduce:
            maps = tf.reduce_mean(maps, axis=-1)
        if self.discard_negative_values:
            maps = tf.nn.relu(maps)
        return maps 
    
    def _filter(self, x: GraphTensor) -> GraphTensor:
        mask = tf.cast(getattr(x, _residue_node_indicator), tf.bool)
        mask = (mask == False if self.node_level else mask)
        return tf.boolean_mask(x, mask, axis='node')