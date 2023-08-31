import tensorflow as tf

import keras
from keras import layers

from typing import Optional
from typing import List
from typing import Any

from molgraph.internal import register_keras_serializable 

from molgraph.tensors.graph_tensor import GraphTensor

from molgraph.layers.preprocessing.masking import FeatureMasking
from molgraph.layers.preprocessing.embedding_lookup import EmbeddingLookup
from molgraph.layers.postprocessing.gather_incident import GatherIncident


@register_keras_serializable(package='molgraph')
class GraphMasking(keras.Model):
    
    '''Masked Graph Modeling (MGM) inspired by Masked Language Modeling (MLM).
    
    See e.g. Hu et al. [#]_ and Devlin et al. [#]_.
    
    The encoder part of the MGM model is the model to be finetuned
    for downstream modeling. In this MGM model, masking layer(s) 
    will be prepended to randomly mask out node and/or edge features 
    of a certain rate. Furthermore, a classification layer is appended 
    to the encoder to produce multi-class predictions; with the objective 
    to predict the masked node and/or edge features. 
    
    Currently, this MGM model only support Tokenized molecular graphs
    which are embedded via `EmbeddingLookup`. A Featurized molecular 
    graph (without `EmbeddingLookup`) is less straight forward to implement, 
    though it could be implemented by e.g. creating random vectors for each 
    masked node/edge.
    
    Example usage:

    >>> # Replace this graph_tensor with a large dataset of graphs
    >>> graph_tensor = molgraph.GraphTensor(
    ...     sizes=[5],
    ...     edge_src=[1, 4, 0, 2, 3, 1, 1, 0],
    ...     edge_dst=[0, 0, 1, 1, 1, 2, 3, 4],
    ...     node_feature=['Sym:C|Hyb:SP3', 'Sym:C|Hyb:SP2', 'Sym:O|Hyb:SP2',
    ...                   'Sym:O|Hyb:SP2', 'Sym:N|Hyb:SP3'],
    ...     edge_feature=['BonTyp:SINGLE|Rot:1', 'BonTyp:SINGLE|Rot:0',
    ...                   'BonTyp:SINGLE|Rot:1', 'BonTyp:DOUBLE|Rot:0',
    ...                   'BonTyp:SINGLE|Rot:0', 'BonTyp:DOUBLE|Rot:0',
    ...                   'BonTyp:SINGLE|Rot:0', 'BonTyp:SINGLE|Rot:0'],
    ... )
    >>> node_embedding = molgraph.layers.NodeEmbeddingLookup(
    ...    32, mask_token='[MASK]')
    >>> edge_embedding = molgraph.layers.EdgeEmbeddingLookup(
    ...    32, mask_token='[MASK]')
    >>> node_embedding.adapt(graph_tensor)
    >>> edge_embedding.adapt(graph_tensor)
    >>> graph_transformer_encoder = tf.keras.Sequential([
    ...     node_embedding,
    ...     edge_embedding,
    ...     molgraph.layers.GTConv(units=32),
    ...     molgraph.layers.GTConv(units=32),
    ...     molgraph.layers.GTConv(units=32),
    ... ])
    >>> pretraining_model = molgraph.models.GraphMasking(
    ...     graph_transformer_encoder, 
    ...     node_feature_masking_rate=0.15, 
    ...     edge_feature_masking_rate=0.15
    ... )
    >>> pretraining_model.compile(
    ...     optimizer=tf.keras.optimizers.Adam(1e-4), 
    ...     metrics=[
    ...         tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_acc'),
    ...     ]
    ... )
    >>> _ = pretraining_model.fit(graph_tensor, epochs=1, verbose=0)
    >>> metric_values = pretraining_model.evaluate(graph_tensor, verbose=0)
    >>> graph_transformer_encoder.save( # doctest: +SKIP
    ...     '/tmp/my_pretrained_encoder_model'
    ... ) 
    >>> loaded_encoder = tf.saved_model.load( # doctest: +SKIP
    ...     '/tmp/my_pretrained_encoder_model'
    ... )
    
    Args:
        encoder (tf.keras.Model):
            The model to be pretrained and used for downstream tasks. The 
            first layer(s) of the model should embed (via `EmbeddingLookup`) 
            (later on, possibly masked) node/edge embeddings from tokenized 
            molecular graphs (obtain via `Tokenizer`).
        node_feature_decoder (None, tf.keras.layers.Layer):
            Optionally supply a decoder which turns the node embeddings 
            into a softmaxed prediction. If None, a linear transdformation
            followed by a softmax activation will be used. Default to None.
        edge_feature_decoder (None, tf.keras.layers.Layer):
            Optionally supply a decoder which turns the edge embeddings 
            into a softmaxed prediction. If None, a linear transdformation
            followed by a softmax activation will be used. Default to None.
        node_feature_masking_rate (None, float):
            The rate at which node features should be masked. If None, or
            0.0, no masking will be performed on the node features. Default
            to 0.15.
        edge_feature_masking_rate (None, float):
            The rate at which edge features should be masked. If None, or
            0.0, no masking will be performed on the edge features. Default
            to 0.15.
    
    References:
        .. [#] https://arxiv.org/pdf/1905.12265.pdf
        .. [#] https://arxiv.org/pdf/1810.04805.pdf
    '''
    
    def __init__(
        self, 
        encoder: keras.Model, 
        node_feature_decoder: Optional[layers.Layer] = None,
        edge_feature_decoder: Optional[layers.Layer] = None,
        node_feature_masking_rate: Optional[float] = 0.15,
        edge_feature_masking_rate: Optional[float] = 0.15,
        name: Optional[str] = 'MaskedGraphModeling',
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        
        self.encoder = encoder

        self.node_feature_masking_rate = node_feature_masking_rate or 0.0
        self.edge_feature_masking_rate = edge_feature_masking_rate or 0.0

        self.node_feature_decoder = node_feature_decoder
        self.edge_feature_decoder = edge_feature_decoder

        for layer in self.encoder.layers:
            
            if isinstance(layer, EmbeddingLookup):
                # feature_type is either "node_feature" or "edge_feature"
                feature_type: str = layer.feature
                
                if (
                    feature_type == 'node_feature' and 
                    node_feature_masking_rate > 0.0
                ):
                    self.node_feature_masking_layer = FeatureMasking(
                        feature=feature_type, 
                        rate=self.node_feature_masking_rate, 
                        mask_token=layer.mask_token)
                    
                    self.lookup_node_feature_label = layer.lookup_table.lookup

                    if self.node_feature_decoder is None:
                        self.node_feature_decoder = layers.Dense(
                            units=layer.vocabulary_size()-1, 
                            activation='softmax')
                        
                elif (
                    feature_type == 'edge_feature' and 
                    edge_feature_masking_rate > 0.0
                ):
                    self.edge_feature_masking_layer = FeatureMasking(
                        feature=feature_type, 
                        rate=self.edge_feature_masking_rate, 
                        mask_token=layer.mask_token)
                                    
                    self.lookup_edge_feature_label = layer.lookup_table.lookup

                    if self.edge_feature_decoder is None:
                        self.edge_feature_decoder = layers.Dense(
                            units=layer.vocabulary_size()-1, 
                            activation='softmax')
                        
                    self.gather_incident = GatherIncident()
            
    def call(
        self, 
        tensor: GraphTensor, 
        training: Optional[bool] = None
    ) -> GraphTensor:
        return self.encoder(tensor, training=training)
    
    def _call(
        self, 
        tensor: GraphTensor,
        training: Optional[bool] = None,    
    ) -> GraphTensor:
        
        if isinstance(tensor.node_feature, tf.RaggedTensor):
            tensor = tensor.merge()
        
        new_data = {}

        if self.node_feature_masking_rate > 0.0:
            node_feature = tensor.node_feature
            tensor = self.node_feature_masking_layer(tensor)
            mask_token = self.node_feature_masking_layer.mask_token
            node_feature_mask = tf.where(
                tensor.node_feature == mask_token, True, False)
            new_data['node_feature_mask'] = node_feature_mask 
            new_data['node_feature_label'] = self.lookup_node_feature_label(
                node_feature) - 1
        
        if self.edge_feature_masking_rate > 0.0:
            edge_feature = tensor.edge_feature
            tensor = self.edge_feature_masking_layer(tensor)
            mask_token = self.edge_feature_masking_layer.mask_token
            edge_feature_mask = tf.where(
                tensor.edge_feature == mask_token, True, False)
            new_data['edge_feature_mask'] = edge_feature_mask 
            new_data['edge_feature_label'] = self.lookup_edge_feature_label(
                edge_feature) - 1

        tensor = self(tensor, training=training)

        return tensor.update(new_data)
    
    def train_step(self, tensor: GraphTensor) -> dict:
        
        with tf.GradientTape() as tape:

            tensor = self._call(tensor, training=True)

            loss = 0.0

            if self.node_feature_masking_rate > 0.0:
                node_loss, (node_true, node_pred) = self._node_feature_loss(
                    tensor)
                loss += node_loss

            if self.edge_feature_masking_rate > 0.0:
                edge_loss, (edge_true, edge_pred) = self._edge_feature_loss(
                    tensor)
                loss += edge_loss

        variables = self.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        if self.node_feature_masking_rate > 0.0:
            self.node_feature_loss_tracker.update_state(node_loss)
            for metric in self.node_metrics:
                metric.update_state(node_true, node_pred)

        if self.edge_feature_masking_rate > 0.0:
            self.edge_feature_loss_tracker.update_state(edge_loss)
            for metric in self.edge_metrics:
                metric.update_state(edge_true, edge_pred)

        return {metric.name: metric.result() for metric in self.metrics}
    
    def test_step(self, tensor: GraphTensor) -> dict:

        tensor = self._call(tensor, training=False)

        if self.node_feature_masking_rate > 0.0:
            node_loss, (node_true, node_pred) = self._node_feature_loss(
                tensor)

        if self.edge_feature_masking_rate > 0.0:
            edge_loss, (edge_true, edge_pred) = self._edge_feature_loss(
                tensor)
            
        if self.node_feature_masking_rate > 0.0:
            self.node_feature_loss_tracker.update_state(node_loss)
            for metric in self.node_metrics:
                metric.update_state(node_true, node_pred)

        if self.edge_feature_masking_rate > 0.0:
            self.edge_feature_loss_tracker.update_state(edge_loss)
            for metric in self.edge_metrics:
                metric.update_state(edge_true, edge_pred)

        return {metric.name: metric.result() for metric in self.metrics}
       
    def predict_step(self, tensor: GraphTensor) -> GraphTensor:
        return self(tensor, training=False)
    
    def _node_feature_loss(
        self, 
        tensor: GraphTensor
    ) -> float:
        y_pred = self._predict_node_features(tensor)
        y_true = tf.boolean_mask(
            tensor.node_feature_label, tensor.node_feature_mask)
        loss = self.loss_fn(y_true, y_pred) * self._node_feature_loss_weight
        return loss, (y_true, y_pred)
    
    def _edge_feature_loss(
        self, 
        tensor: GraphTensor
    ) -> float:
        y_pred = self._predict_edge_features(tensor)
        y_true = tf.boolean_mask(
            tensor.edge_feature_label, tensor.edge_feature_mask)
        loss = self.loss_fn(y_true, y_pred) * self._edge_feature_loss_weight
        return loss, (y_true, y_pred)
    
    def _predict_node_features(
        self, 
        tensor: GraphTensor
    ) -> tf.Tensor:
        node_feature = tf.boolean_mask(
            tensor.node_feature, tensor.node_feature_mask)
        y_pred = self.node_feature_decoder(node_feature)
        return y_pred

    def _predict_edge_features(
        self, 
        tensor: GraphTensor
    ) -> tf.Tensor:
        tensor = tf.boolean_mask(
            tensor, tensor.edge_feature_mask, axis='edge')
        edge_feature = self.gather_incident(tensor)
        y_pred = self.edge_feature_decoder(edge_feature)
        return y_pred
    
    def compile(
        self, 
        optimizer: keras.optimizers.Optimizer, 
        loss: Optional[keras.losses.Loss] = None, 
        metrics: Optional[List[keras.metrics.Metric]] = None, 
        node_feature_loss_weight: Optional[float] = None,
        edge_feature_loss_weight: Optional[float] = None, 
        *args, 
        **kwargs
    ):
        '''Configures the model for training.
        
        Args:
            optimizer (tf.keras.optimizers.Optimizer):
                The optimizer to use for training.
            loss (None, tf.keras.losses.Loss):
                The loss function to use. If None, 
                `tf.keras.losses.SparseCategoricalCrossentropy` will be used.
                If a custom loss function is used, make sure it deals with 
                sparse labels (i.e. integer encoding and not one-hot encoding). 
                Default to None.
            metrics (None, list[tf.keras.metrics.Metric]):
                The metrics to use. Default to None.
            node_feature_loss_weight (None, float):
                The weight to be applied to the node feature prediction loss.
                If None, the node feature loss will be multiplied by 1. 
                Default to None.
            edge_feature_loss_weight (None, float):
                THe wieght to be applied to the edge feature prediction loss.
                If None, the edge feature loss will be multiplied by 1. 
                Default to None.
            *args:
                See tf.keras.Model.compile documentation.
            **kwargs:
                See tf.keras.Model.compile documentation.
        '''
        super().compile(
            optimizer=optimizer, 
            loss=None, 
            metrics=None, 
            *args, 
            **kwargs)

        self.loss_fn = (
            keras.losses.SparseCategoricalCrossentropy(name='sparse_cce') 
            if loss is None else loss)
   
        metrics = [] if metrics is None else metrics

        if self.node_feature_masking_rate > 0.0:
            self.node_metrics = []
            self._node_feature_loss_weight = (
                node_feature_loss_weight or 1.0)
            self.node_feature_loss_tracker = keras.metrics.Mean(
                name='node_' + self.loss_fn.name)
            for metric in metrics:
                metric_config = metric.get_config()
                metric_config['name'] = 'node_' + metric_config['name']
                self.node_metrics.append(metric.from_config(metric_config))

        if self.edge_feature_masking_rate > 0.0:
            self.edge_metrics = []
            self._edge_feature_loss_weight = (
                edge_feature_loss_weight or 1.0)
            self.edge_feature_loss_tracker = keras.metrics.Mean(
                name='edge_' + self.loss_fn.name)
            for metric in metrics:
                metric_config = metric.get_config()
                metric_config['name'] = 'edge_' + metric_config['name']
                self.edge_metrics.append(metric.from_config(metric_config))

    @property
    def metrics(self) -> List[keras.metrics.Metric]:
        metrics = []
        if self.node_feature_masking_rate > 0.0:
            metrics.append(self.node_feature_loss_tracker)
        if self.edge_feature_masking_rate > 0.0:
            metrics.append(self.edge_feature_loss_tracker)
        return metrics + self.node_metrics + self.edge_metrics
    
    def fit(
        self, 
        x, 
        y: Any = None, 
        batch_size: Optional[int] = None, 
        epochs: int = 1, 
        *args, 
        **kwargs
    ):
        '''Trains the autoencoder for a fixed number of epochs.

        Args:
            x (GraphTensor, tf.data.Dataset):
                The input data. Either a `GraphTensor` instance or a 
                `tf.data.Dataset` of `GraphTensor`s.
            y (None):
                Target data, which will be ignored as the target can
                be obtained from `x`.
            batch_size (int, None):
                Number of samples per gradient update. If `None`,
                32 will be used. Default to `None`.
            epochs (int):
                Number of iterations over all subgraphs (molecular graphs)
                of the `GraphTensor` instance or `tf.data.Dataset` instance.
                Default to 1.
            *args: 
                See tf.keras.Model.fit documentaton.
            **kwargs:
                See tf.keras.Model.fit documentation.
        
        Returns:
            A `History` object containing e.g. training loss values
            and metric values.
        '''
        return super().fit(
            x=x, y=None, batch_size=batch_size, epochs=epochs, *args, **kwargs)

    def evaluate(
        self, 
        x, 
        y: Any = None, 
        batch_size: Optional[int] = None, 
        *args, 
        **kwargs
    ):
        '''Evaluates the autoencoder.

        Args:
            x (GraphTensor, tf.data.Dataset):
                The input data; either a `GraphTensor` instance or a 
                `tf.data.Dataset` of `GraphTensor`s.
            y (None):
                Target data; which will be ignored as the target can
                be obtained from `x`.
            batch_size (int, None):
                Number of samples per batch of computation. If `None`,
                32 will be used. Default to `None`.
            *args: 
                See tf.keras.Model.evaluate documentaton.
            **kwargs:
                See tf.keras.Model.evaluate documentation.
        
        Returns:
            Average loss values (e.g. reconstruction loss and kl loss).
        '''
        return super().evaluate(
            x=x, y=None, batch_size=batch_size, *args, **kwargs)
    
    def predict(
        self, 
        x, 
        batch_size: Optional[int] = None, 
        *args, 
        **kwargs
    ):
        '''Generates outputs of the autoencoder.

        Args:
            x (GraphTensor, tf.data.Dataset):
                The input data; either a `GraphTensor` instance or a 
                `tf.data.Dataset` of `GraphTensor`s.
            batch_size (int, None):
                Number of samples per batch of computation. If `None`,
                32 will be used. Default to `None`.
            *args: 
                See tf.keras.Model.evaluate documentaton.
            **kwargs:
                See tf.keras.Model.evaluate documentation.
        
        Returns:
            `tf.Tensor` or `tf.RaggedTensor` of edge scores, corresponding to 
            the inputted `GraphTensor` instance.
        '''
        return super().predict(
            x=x, batch_size=batch_size, *args, **kwargs)
    
    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'encoder': layers.serialize(self.encoder),
            'node_feature_decoder': layers.serialize(self.node_feature_decoder),
            'edge_feature_decoder': layers.serialize(self.edge_feature_decoder),
            'node_feature_masking_rate': self.node_feature_masking_rate,
            'edge_feature_masking_rate': self.edge_feature_masking_rate,
        })
        return config