import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from typing import Tuple

from molgraph.tensors.graph_tensor import GraphTensor



@keras.utils.register_keras_serializable(package='molgraph')
class SetGatherReadout(layers.Layer):

    '''Set-to-set layer for graph readout.

    Implementation based on Gilmer et al. (2017) [#]_ and Vinyals et al. (2016) [#]_.

    **Example:**

    >>> graph_tensor = molgraph.GraphTensor(
    ...     data={
    ...         'edge_dst': [[0, 1], [0, 0, 1, 1, 2, 2]],
    ...         'edge_src': [[1, 0], [1, 2, 0, 2, 1, 0]],
    ...         'node_feature': [
    ...             [[1.0, 0.0], [1.0, 0.0]],
    ...             [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    ...         ],
    ...     }
    ... )
    >>> model = tf.keras.Sequential([
    ...     tf.keras.layers.Input(type_spec=graph_tensor.unspecific_spec),
    ...     # molgraph.layers.GCNConv(4),
    ...     molgraph.layers.SetGatherReadout()
    ... ])
    >>> # Note: SetGatherReadout increases output dim (from 2 to 4)
    >>> model.output_shape
    (None, 4)

    Args:
        steps (int):
            Number of LSTM steps. Default to 8.

    References:
        .. [#] https://arxiv.org/pdf/1704.01212.pdf
        .. [#] https://arxiv.org/pdf/1511.06391.pdf
    '''

    def __init__(self, steps: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.steps = steps
        self.lstm_cell = NoInputLSTMCell()

    def call(self, tensor: GraphTensor) -> tf.Tensor:
        '''Defines the computation from inputs to outputs.

        This method should not be called directly, but indirectly
        via ``__call__()``. Upon first call, the layer is automatically
        built via ``build()``.

        Args:
            tensor (GraphTensor):
                Input to the layer.

        Returns:
            A ``tf.Tensor`` or `tf.RaggedTensor` based on the ``node_feature``
            field of the inputted ``GraphTensor``.
        '''
        if isinstance(tensor.node_feature, tf.RaggedTensor):
            tensor = tensor.merge()

        node_dim = tensor.node_feature.shape[-1]
        graph_indicator = tf.squeeze(tensor.graph_indicator)
        node_feature = tensor.node_feature

        # Obtain the number of molecules in the batch
        num_graphs = tf.reduce_max(graph_indicator) + 1

        # Initialize states
        memory_state = tf.zeros((num_graphs, node_dim))
        carry_state = tf.zeros((num_graphs, node_dim))

        # Perform a number of lstm steps (via the set-to-set procedure)
        for i in range(self.steps):

            # Expand carry state to match node_feature
            carry_state_expanded = tf.gather(carry_state, graph_indicator)

            # Perform a linear transformation followed by reduction
            node_feature_reduced = tf.reduce_sum(
                node_feature * carry_state_expanded, axis=1)

            # Split into parts correspoding to each graph in the batch,
            # and compute normalized attention coefficients via softmax
            def body(i, attention_coef):
                node_feature_partitioned = tf.gather(
                    node_feature_reduced, tf.where(graph_indicator == i)[:, 0])
                attention_coef = attention_coef.write(
                    tf.cast(i, tf.int32), tf.nn.softmax(node_feature_partitioned))
                return tf.add(i, 1), attention_coef

            def cond(i, _):
                return tf.less(i, num_graphs)

            i = tf.constant(0, dtype=graph_indicator.dtype)
            attention_coef = tf.TensorArray(
                tf.float32, size=0, dynamic_size=True, infer_shape=False)
            _, attention_coef = tf.while_loop(cond, body, [i, attention_coef])
            attention_coef = tf.reshape(attention_coef.concat(), (-1, 1))

            # Apply attention to node features, and sum based on graph_indicator
            attention_readout = tf.math.segment_sum(
                attention_coef * node_feature, graph_indicator)

            # Concatenate (previous) carry_state and attention readout
            carry_state_evolved = tf.concat([
                carry_state, attention_readout], axis=1)

            # Perform a LSTM step (with only a memory state and carry state)
            memory_state, carry_state = self.lstm_cell([
                memory_state, carry_state_evolved])

        return carry_state_evolved

    def compute_output_shape(self, input_shape):
        input_shape[-1] = int(input_shape[-1] * 2)
        if input_shape[0] is None and input_shape[1] is not None:
            # input_shape corresponds to a tf.Tensor
            return input_shape
        # input_shape corresponds to a tf.RaggedTensor
        return input_shape[1:]

    def get_config(self):
        base_config = super().get_config()
        config = {
            'steps': self.steps,
        }
        base_config.update(config)
        return base_config


@keras.utils.register_keras_serializable(package='molgraph')
class NoInputLSTMCell(layers.Layer):

    'Custom LSTM Cell that takes no input'

    def build(self, input_shape):
        memory_state_dim = input_shape[0][-1]
        carry_state_dim = input_shape[1][-1]
        self.recurrent_kernel = self.add_weight(
            shape=(carry_state_dim, memory_state_dim * 4),
            trainable=True,
            initializer="glorot_uniform",
            name='recurrent_kernel'
        )
        self.bias = self.add_weight(
            shape=(memory_state_dim * 4),
            trainable=True,
            initializer="zeros",
            name='bias'
        )
        self.built = True

    def call(
        self,
        inputs: Tuple[tf.Tensor, tf.Tensor]
    ) -> Tuple[tf.Tensor, tf.Tensor]:

        # Unpack states
        memory_state, carry_state = inputs

        # Perform linear transformation on carry_state
        z = tf.matmul(carry_state, self.recurrent_kernel) + self.bias

        # Split transformed carry_state into four units (gates/states)
        update_gate, forget_gate, memory_state_candidate, output_gate = tf.split(
            z, num_or_size_splits=4, axis=1
        )

        # Apply non-linearity to all units
        update_gate = tf.nn.sigmoid(update_gate)
        forget_gate = tf.nn.sigmoid(forget_gate)
        output_gate = tf.nn.sigmoid(output_gate)
        memory_state_candidate = tf.nn.tanh(memory_state_candidate)

        # Forget and update memory state
        memory_state = forget_gate * memory_state + update_gate * memory_state_candidate

        # Update carry state
        carry_state = output_gate * tf.nn.tanh(memory_state)

        # Return (updated) memory state and carry state
        return memory_state, carry_state
