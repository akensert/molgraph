import tensorflow as tf
from tensorflow import keras
from typing import Dict
from typing import Tuple

from molgraph.models.base import BaseModel
from molgraph.layers.postprocessing.dot_product_incident import DotProductIncident
from molgraph.tensors.graph_tensor import GraphTensor



class LinkPredictionPretrainer(BaseModel):

    def __init__(
        self,
        gconv_model: keras.Model,
        k: int = 1,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.gconv_model = gconv_model
        self.dot_incident = DotProductIncident(partition=False)
        self.k = k

    def train_step(self, data: GraphTensor) -> Dict[str, float]:

        with tf.GradientTape() as tape:

            score_pos, score_neg = self(data[0], training=True)

            loss = self.compiled_loss(
                score_pos, score_neg, regularization_losses=self.losses)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(score_pos, score_neg)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data: GraphTensor) -> Dict[str, float]:
        score_pos, score_neg = self(data[0], training=False)
        self.compiled_loss(
            score_pos, score_neg, regularization_losses=self.losses)
        self.compiled_metrics.update_state(score_pos, score_neg)
        return {m.name: m.result() for m in self.metrics}

    def call(self, tensor: GraphTensor) -> Tuple[tf.Tensor, tf.Tensor]:
        tensor = self.gconv_model(tensor)
        edge_score_pos = self.dot_incident(tensor).edge_score
        edge_score_neg = self.dot_incident(
            sample_negative_graph(tensor, self.k)).edge_score

        edge_score_pos = tf.reshape(edge_score_pos, (-1, 1))
        edge_score_neg = tf.reshape(edge_score_neg, (-1, 1))
        return edge_score_pos, edge_score_neg


def sample_negative_graph(tensor: GraphTensor, k: int) -> GraphTensor:
    """Replaces edges in the input tensor by shuffling the destination edges.

    Also accepts a parameter `k` which defines how many more negative edges will
    exist compared to positive edges (i.e. positive/negative edges ratio)
    """
    edge_dst = tf.repeat(tensor.edge_dst, k)
    edge_src = tf.repeat(tensor.edge_src, k)
    edge_dst = tf.random.shuffle(edge_dst)
    return tensor.replace({'edge_dst': edge_dst, 'edge_src': edge_src})
