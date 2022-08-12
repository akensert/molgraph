import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict
from typing import Tuple
from typing import Optional

from molgraph.models.base import BaseModel
from molgraph.tensors.graph_tensor import GraphTensor
from molgraph.layers.preprocessing.projection import FeatureProjection


class MaskedGraphPretrainer(BaseModel):

    def __init__(self, model: keras.Model, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.mask_layer = _FeatureMasking(
            'node_feature', rate, self.model.layers[0].lookups)
        self.projection = FeatureProjection(
            feature='node_feature',
            units=self.model.layers[0].get_total_vocabulary_size(include_mask=False),
            activation='sigmoid')

    def train_step(self, tensor):

        tensor = tensor[0]

        with tf.GradientTape() as tape:
            tensor = self(tensor, training=True)
            y_pred = tf.boolean_mask(
                tensor.node_feature, tensor._node_feature_mask)
            y_true = tensor._node_feature_label
            loss = self.compiled_loss(
                y_true, y_pred, regularization_losses=self.losses)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y_true, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, tensor):
        tensor = tensor[0]
        tensor = self(tensor, training=False)
        y_pred = tf.boolean_mask(
                tensor.node_feature, tensor._node_feature_mask)
        y_true = tensor._node_feature_label
        self.compiled_loss(
            y_true, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y_true, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, tensor):
        tensor = tensor[0]
        tensor = self(tensor, training=False)
        return tensor

    def call(self, tensor: GraphTensor) -> GraphTensor:
        tensor = self.mask_layer(tensor)
        for layer in self.model.layers:
            if 'Readout' in layer.__class__.__name__:
                break
            tensor = layer(tensor)
        return self.projection(tensor)


class _FeatureMasking(layers.Layer):
    def __init__(self, feature, rate, lookups, **kwargs):
        super().__init__(**kwargs)

        self.feature = feature
        self.rate = rate
        self.lookups = lookups
        self.depths = tf.nest.map_structure(
            lambda lookup: lookup.vocabulary_size() - 1,
            self.lookups)

    def call(self, tensor, training=False):
        feature = getattr(tensor, self.feature)

        shape = tf.shape(tf.nest.flatten(feature)[0])[:1]
        random_values = tf.random.uniform(shape)
        mask = random_values > (1 - self.rate)

        label = tf.nest.map_structure(
            lambda x: tf.boolean_mask(x, mask), feature)

        label = tf.nest.map_structure(
            lambda x, lookup, depth: tf.one_hot(lookup(x) - 1, depth=depth),
            label, self.lookups, self.depths)

        label = tf.concat(tf.nest.flatten(label), axis=1)

        tensor = tensor.add({
            f'_{self.feature}_mask': mask,
            f'_{self.feature}_label': label})

        feature = tf.nest.map_structure(
            lambda x: tf.where(mask, tf.constant(0, dtype=tf.int32), x), feature)

        return tensor.replace({self.feature: feature})
