import tensorflow as tf
from tensorflow import keras
from keras import layers

from typing import Optional
from typing import Any

from molgraph.internal import register_keras_serializable 

from molgraph.tensors.graph_tensor import GraphTensor

from molgraph.layers.postprocessing.dot_product_incident import DotProductIncident


# TODO: Improve negative graph sampler. Need to distinguish separate but
#       equivalent atoms.

    
@register_keras_serializable(package='molgraph')
class GraphAutoEncoder(keras.Model):
    '''Graph AutoEncoder (GAE) based on Kipf and Welling [#]_.
    
    Args:
        encoder (tf.keras.layers.Layer):
            The encoder part of the autoencoder. The encoder could be
            any of the graph neural neural network layers provided by
            molgraph; e.g., ``GCNConv``, ``GINConv`` or ``GATv2Conv``.
        decoder (tf.keras.layers.Layer):
            The decoder part of the autoencoder. If None is passed,
            the decoder used is ``DotProductIncident``. Default to None.
        negative_graph_sampler (callable):
            A function which samples negative graphs. It takes as input
            the current GraphTensor instance and produces a new
            GraphTensor instance with negative edges. If None, 
            ``NegativeGraphSampler`` will be used. Default to None.
        balanced_class_weighting (bool):
            Whether balanced class weighting should be performed. For
            instance, if there are twice as many negative edges, should
            each negative edge be weighted 0.5 for the loss?
    
    Example usage:

    >>> # Replace this graph_tensor with a large dataset of graphs
    >>> # Also accepts featurized graph tensors (via `chemistry.Featurizer`)
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
    >>> # Obtain GraphAutoEncoder model
    >>> encoder = tf.keras.Sequential([
    ...     tf.keras.layers.Input(type_spec=graph_tensor.spec),
    ...     node_embedding,
    ...     edge_embedding,
    ...     molgraph.layers.GATv2Conv(128),
    ...     molgraph.layers.GATv2Conv(128),
    ...     molgraph.layers.GATv2Conv(128),
    ... ])
    >>> decoder = molgraph.layers.DotProductIncident(normalize=True)
    >>> gae = molgraph.models.GraphAutoEncoder(encoder, decoder)
    >>> gae.compile('adam')
    >>> _ = gae.fit(graph_tensor, batch_size=32, epochs=50, verbose=0)
    >>> reconstruction_loss = gae.evaluate(graph_tensor, verbose=0)
    >>> encoder.save( # doctest: +SKIP
    ...     '/tmp/my_pretrained_encoder_model'
    ... ) 
    >>> loaded_encoder = tf.saved_model.load( # doctest: +SKIP
    ...     '/tmp/my_pretrained_encoder_model'
    ... )

    References:
        .. [#] https://arxiv.org/pdf/1611.07308.pdf
    '''
    def __init__(
        self, 
        encoder: tf.keras.layers.Layer, 
        decoder: Optional[tf.keras.layers.Layer] = None, 
        negative_graph_sampler: Optional[tf.keras.layers.Layer] = None,
        balanced_class_weighting: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = (
            decoder if decoder is not None 
            else DotProductIncident(normalize=True)
        )
        self.negative_graph_sampler = (
            negative_graph_sampler if negative_graph_sampler is not None 
            else NegativeGraphSampler(1)
        )
        self.balanced_class_weighting = balanced_class_weighting
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="rec_loss")

    @property
    def metrics(self):
        return [
            self.reconstruction_loss_tracker
        ]
        
    def train_step(self, tensor: GraphTensor):
        
        if tensor.is_ragged():
            tensor = tensor.merge()
            
        with tf.GradientTape() as tape:
            encoded = self(tensor, training=True)
            encoded_neg = self.negative_graph_sampler(encoded)
            decoded_neg = self.decoder(encoded_neg)
            decoded_pos = self.decoder(encoded)
            
            if self.balanced_class_weighting:
                sample_weight = _compute_balanced_class_weighting(
                    decoded_pos.edge_score, decoded_neg.edge_score)
            else:
                sample_weight = 1.
            
            if self.loss_fn is None:
                scores = tf.concat([
                    decoded_pos.edge_score * -1., 
                    decoded_neg.edge_score], axis=0)
                reconstruction_loss = scores * sample_weight
                reconstruction_loss = tf.reduce_mean(reconstruction_loss)
            else:
                reconstruction_loss = self.loss_fn(
                    decoded_pos.edge_score, 
                    decoded_neg.edge_score, 
                    sample_weight=sample_weight)
                
            reg_loss = sum(self.losses)

            loss = reconstruction_loss + reg_loss
            
        trainable_variables = self.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        self.reconstruction_loss_tracker.update_state(reconstruction_loss)

        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, tensor: GraphTensor):
        
        if tensor.is_ragged():
            tensor = tensor.merge()

        encoded = self(tensor, training=False)
        encoded_neg = self.negative_graph_sampler(encoded)
        decoded_neg = self.decoder(encoded_neg)
        decoded_pos = self.decoder(encoded)
        
        if self.balanced_class_weighting:
            sample_weight = _compute_balanced_class_weighting(
                decoded_pos.edge_score, decoded_neg.edge_score)
        else:
            sample_weight = 1.

        if self.loss_fn is None:
            scores = tf.concat([
                decoded_pos.edge_score * -1., 
                decoded_neg.edge_score], axis=0)
            reconstruction_loss = scores * sample_weight
            reconstruction_loss = tf.reduce_mean(reconstruction_loss)
        else:
            reconstruction_loss = self.loss_fn(
                decoded_pos.edge_score, 
                decoded_neg.edge_score, 
                sample_weight=sample_weight)
        
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)

        return {m.name: m.result() for m in self.metrics}
    
    def predict_step(self, tensor: GraphTensor) -> GraphTensor:
        
        tensor_orig = tensor
        if tensor.is_ragged():
            tensor = tensor.merge()

        encoded = self(tensor, training=False)
        decoded = self.decoder(encoded)
        return tensor_orig.update({
            'edge_score': decoded.edge_score}).edge_score
        
    def call(self, tensor: GraphTensor) -> GraphTensor:
        return self.encoder(tensor)

    def compile(self, optimizer, loss=None, *args, **kwargs):
        '''Configures the model for training.
        
        Args:
            optimizer (tf.keras.optimizers.Optimizer):
                The optimizer to use for training.
            loss (None, tf.keras.losses.Loss):
                The loss function to use. If None, a default loss function
                will be used. This loss function simply just tried to 
                maximize the values of the positive edges and minimize the
                values of the negative edges. If a custom loss function is used,
                be aware that the inputs to the loss functions are: positive 
                edge scores (`y_true` ) and negative edge scores (`y_pred`);
                both resulting from `molgraph.layers.DotProductIncident(...)`.
                Default to None.
            metrics (None):
                This argument will be ignored (at least for now).
            *args:
                See tf.keras.Model.compile documentation.
            **kwargs:
                See tf.keras.Model.compile documentation.
        '''
        super().compile(
            optimizer=optimizer, loss=None, metrics=None, *args, **kwargs)
        
        if loss is not None:
            self.loss_fn = loss
        else:
            self.loss_fn = None

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

    def get_config(self):
        config = super().get_config()
        config.update({
            'encoder': layers.serialize(self.encoder),
            'decoder': layers.serialize(self.decoder),
            'negative_graph_sampler': layers.serialize(
                self.negative_graph_sampler),
            'balanced_class_weighting': self.balanced_class_weighting,
        })
        return config
    

# TODO: instead of beta initial/end/incr, pass a scheduler?
@register_keras_serializable(package='molgraph')
class GraphVariationalAutoEncoder(GraphAutoEncoder):
    '''Graph Variational AutoEncoder (GAE) based on Kipf and Welling [#]_.
    
    Args:
        encoder (tf.keras.layers.Layer):
            The encoder part of the autoencoder. The encoder could be
            any of the graph neural neural network layers provided by
            molgraph; e.g., ``GCNConv``, ``GINConv`` or ``GATv2Conv``.
        decoder (tf.keras.layers.Layer):
            The decoder part of the autoencoder. If None is passed,
            the decoder used is ``DotProductIncident``. Default to None.
        negative_graph_sampler (callable):
            A function which samples negative graphs. It takes as input
            the current GraphTensor instance and produces a new
            GraphTensor instance with negative edges. If None, 
            ``NegativeGraphSampler`` will be used. Default to None.
        balanced_class_weighting (bool):
            Whether balanced class weighting should be performed. For
            instance, if there are twice as many negative edges, should
            each negative edge be weighted 0.5 for the loss?
        beta_initial (float):
            Initial beta value (which is multiplied with the kl loss). 
            Default to 0.0.
        beta_end (float):
            End beta value. Default to 0.1.
        beta_incr (float):
            The increment rate of the beta value (updated each train step).
            Default to 1e-6.
    
    Example usage:

    >>> # Replace this graph_tensor with a large dataset of graphs
    >>> # Also accepts featurized graph tensors (via `chemistry.Featurizer`)
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
    >>> graph_tensor = graph_tensor.separate()
    >>> node_embedding = molgraph.layers.NodeEmbeddingLookup(
    ...    32, mask_token='[MASK]')
    >>> edge_embedding = molgraph.layers.EdgeEmbeddingLookup(
    ...    32, mask_token='[MASK]')
    >>> node_embedding.adapt(graph_tensor)
    >>> edge_embedding.adapt(graph_tensor)
    >>> # Obtain the encoder of GVAE
    >>> encoder_inputs = tf.keras.layers.Input(type_spec=graph_tensor.spec)
    >>> encoder_x = node_embedding(encoder_inputs)
    >>> encoder_x = edge_embedding(encoder_x)
    >>> encoder_x = molgraph.layers.GATv2Conv(128, name='shared_conv')(encoder_x)
    >>> encoder_x_mean = molgraph.layers.GATv2Conv(128, name='loc_conv')(encoder_x)
    >>> encoder_x_log_var = molgraph.layers.GATv2Conv(128, name='log_var_conv')(encoder_x)
    >>> encoder = tf.keras.Model(encoder_inputs, [encoder_x_mean, encoder_x_log_var])
    >>> # Obtain the decoder of GVAE
    >>> decoder = molgraph.layers.DotProductIncident(normalize=True)
    >>> # Obtain, train and evaluate GVAE model
    >>> gvae = molgraph.models.GraphVariationalAutoEncoder(encoder, decoder)
    >>> gvae.compile('adam')
    >>> _ = gvae.fit(graph_tensor, batch_size=32, epochs=50, verbose=0)
    >>> total_loss, rec_loss, kl_loss = gvae.evaluate(graph_tensor, verbose=0)
    >>> encoder.save( # doctest: +SKIP
    ...     '/tmp/my_pretrained_encoder_model'
    ... ) 
    >>> loaded_encoder = tf.saved_model.load( # doctest: +SKIP
    ...     '/tmp/my_pretrained_encoder_model'
    ... )

    References:
        .. [#] https://arxiv.org/pdf/1611.07308.pdf
    '''
    def __init__(
        self, 
        encoder: tf.keras.layers.Layer, 
        decoder: Optional[tf.keras.layers.Layer] = None, 
        negative_graph_sampler: Optional[tf.keras.layers.Layer] = None,
        balanced_class_weighting: bool = False,
        beta_initial: float = 0.00,
        beta_end: float = 0.1,
        beta_incr: float = 1e-6,
        **kwargs
    ):
        super().__init__(
            encoder=encoder, 
            decoder=decoder, 
            negative_graph_sampler=negative_graph_sampler, 
            balanced_class_weighting=balanced_class_weighting
        )
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.beta_initial = beta_initial
        self.beta_end = beta_end
        self.beta_incr = beta_incr
        self.beta = tf.Variable(
            initial_value=beta_initial, 
            dtype=tf.float32, 
            trainable=False)
        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    def train_step(self, tensor: GraphTensor):

        if isinstance(tensor.node_feature, tf.RaggedTensor):
            tensor = tensor.merge()

        with tf.GradientTape() as tape:

            encoded = self(tensor, training=True)
            encoded_neg = self.negative_graph_sampler(encoded)
            decoded_neg = self.decoder(encoded_neg)
            decoded_pos = self.decoder(encoded)
            
            if self.balanced_class_weighting:
                sample_weight = _compute_balanced_class_weighting(
                    decoded_pos.edge_score, decoded_neg.edge_score)
            else:
                sample_weight = 1.

            if self.loss_fn is None:
                scores = tf.concat([
                    decoded_pos.edge_score * -1., 
                    decoded_neg.edge_score], axis=0)
                reconstruction_loss = scores * sample_weight
                reconstruction_loss = tf.reduce_mean(reconstruction_loss)
            else:
                reconstruction_loss = self.loss_fn(
                    decoded_pos.edge_score, 
                    decoded_neg.edge_score, 
                    sample_weight=sample_weight)
            
            kl_loss = self.kl_loss(
                encoded.node_feature_mean, encoded.node_feature_log_var)
            
            reg_loss = sum(self.losses)
            
            total_loss = reconstruction_loss + kl_loss
     
            loss = reconstruction_loss + self.beta * kl_loss + reg_loss
            
        variables = self.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        self.beta.assign(
            tf.minimum(self.beta + self.beta_incr, self.beta_end))
        
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, tensor: GraphTensor):
        
        if isinstance(tensor.node_feature, tf.RaggedTensor):
            tensor = tensor.merge()
            
        encoded = self(tensor, training=False)
        encoded_neg = self.negative_graph_sampler(encoded)
        decoded_neg = self.decoder(encoded_neg)
        decoded_pos = self.decoder(encoded)
        
        if self.balanced_class_weighting:
            sample_weight = _compute_balanced_class_weighting(
                decoded_pos.edge_score, decoded_neg.edge_score)
        else:
            sample_weight = 1.

        if self.loss_fn is None:
            scores = tf.concat([
                decoded_pos.edge_score * -1., 
                decoded_neg.edge_score], axis=0)
            reconstruction_loss = scores * sample_weight
            reconstruction_loss = tf.reduce_mean(reconstruction_loss)
        else:
            reconstruction_loss = self.loss_fn(
                decoded_pos.edge_score, 
                decoded_neg.edge_score, 
                sample_weight=sample_weight)
    
        kl_loss = self.kl_loss(
            encoded.node_feature_mean, encoded.node_feature_log_var)
        
        total_loss = reconstruction_loss + kl_loss
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {m.name: m.result() for m in self.metrics}
        
    def call(self, tensor: GraphTensor, training: bool) -> GraphTensor:
        (tensor_z_mean, tensor_z_log_var) = self.encoder(tensor)
        
        z_mean = tensor_z_mean.node_feature
        z_log_var = tensor_z_log_var.node_feature

        if training:
            z_shape = tf.shape(z_log_var)
            z = z_mean + tf.exp(0.5 * z_log_var) * tf.random.normal(z_shape)
        else:
            z = z_mean
 
        return tensor_z_mean.update({
            'node_feature': z,
            'node_feature_mean': z_mean,
            'node_feature_log_var': z_log_var,
        })
    
    @staticmethod
    def kl_loss(z_mean, z_log_var):
        return tf.reduce_mean(
            tf.reduce_sum(
                -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)),
                axis=1
            )
        )
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'beta_initial': self.beta_initial,
            'beta_end': self.beta_end,
            'beta_incr': self.beta_incr
        })
        return config


@register_keras_serializable(package='molgraph')
class NaiveNegativeGraphSampler(layers.Layer):
    
    '''Samples a negative graphs, or rather, a graph with negative edges.
    
    It is a "naive" implementation as it simply just shuffles `edge_dst` of
    the `GraphTensor` instance, without considering that the resulting
    `edge_dst` and `edge_src` pair could be a positive (correct) edge.
    
    Args:
        k (int):
            Defines how many more negative edges should be computed compared 
            to positive edges (i.e. negative/positive edges ratio). Takes
            on an integer value between [[1, inf]]. If k > 1, more negative 
            than positive examples will be trained on. If k = 1, then the 
            same number of negative and positive examples will be trained
            on. Default to 1.
    '''
    
    def __init__(self, k: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.k = k
    
    def call(self, tensor: GraphTensor) -> GraphTensor:
        edge_dst = tf.repeat(tensor.edge_dst, self.k)
        edge_src = tf.repeat(tensor.edge_src, self.k)
        edge_dst = tf.random.shuffle(edge_dst)
        data = tensor.data.copy()
        data['edge_dst'] = edge_dst
        data['edge_src'] = edge_src
        return tensor.__class__(**data)
    
    def get_config(self):
        config = super().get_config()
        config.update({'k': self.k})
        return config
    

@register_keras_serializable(package='molgraph')
class NegativeGraphSampler(NaiveNegativeGraphSampler):
    
    '''Samples a negative graphs, or rather, a graph with negative edges.
    
    This is a "non-naive" implementation as it makes sure that the original
    (positive) edges do not exist in the set of negative edges.
    
    Args:
        k (int):
            Defines how many more negative edges should be computed compared 
            to positive edges (i.e. negative/positive edges ratio). Takes
            on an integer value between [[1, inf]]. If k > 1, more negative 
            than positive examples will be trained on. If k = 1, then the 
            same number of negative and positive examples will be trained
            on. Default to 1.
    '''
    
    def call(self, tensor: GraphTensor) -> GraphTensor:
        
        # Produce a "keep" mask, which finds all the allowable negative edges 
        mask = tf.logical_and(
            # edge_src_{i} should not match with edge_dst_{j} = edge_dst_{i}
            tensor.edge_dst[:, None] != tensor.edge_dst, 
            # edge_src_{i} should not match with edge_src_{j} = edge_src_{i}
            tensor.edge_src[:, None] != tensor.edge_dst
        )
        mask = tf.gather(
            # segment_min with segment_ids=edge_src allow us to 
            # combine mask of all edge_src_{k} = edge_src_{i}
            tf.math.unsorted_segment_min(
                data=tf.cast(mask, tf.int8), 
                segment_ids=tensor.edge_src, 
                num_segments=tf.shape(tensor.node_feature)[0]
            ),
            tensor.edge_src
        )
        edges = tf.where(tf.cast(mask, tf.bool))
        
        # number of allowable edges is huge, need to mask some out, based on `k`
        num_positives = tf.shape(tensor.edge_src)[0]
        num_negatives = tf.shape(edges)[0]
        ratio = num_negatives // num_positives
        keep_prob = tf.cast(self.k / ratio, tf.float32)
        keep_mask = tf.random.uniform((num_negatives,), dtype=tf.float32) < keep_prob
        
        edges = tf.boolean_mask(edges, keep_mask)

        edge_src_neg = tf.gather(tensor.edge_src, edges[:, 0])
        edge_dst_neg = tf.gather(tensor.edge_dst, edges[:, 1])
        
        data = tensor.data.copy()
        data['edge_src'] = edge_src_neg
        data['edge_dst'] = edge_dst_neg

        return tensor.__class__(**data)
    

def _compute_balanced_class_weighting(edge_score_pos, edge_score_neg):
    num_positives = tf.shape(edge_score_pos)[0] # num positive edges
    num_negatives = tf.shape(edge_score_neg)[0] # num negative edges
    ratio = tf.cast(num_positives / num_negatives, edge_score_pos.dtype)
    sample_weight = tf.concat([[1.], [ratio]], axis=0)
    sample_weight = tf.repeat(sample_weight, [num_positives, num_negatives])
    sample_weight = tf.reshape(sample_weight, [-1, 1])
    return sample_weight