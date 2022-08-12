# import tensorflow as tf
# from tensorflow import keras
# from abc import ABC
# from abc import abstractmethod
#
# from molgraph.tensors.graph_tensor import GraphTensor
#
#
# class BaseModel(keras.Model, ABC):
#
#     def __init__(self, **kwargs):
#         self.tf_function = kwargs.pop('tf_function', True)
#         keras.Model.__init__(self, **kwargs)
#
#     def __call__(self, *args, **kwargs):
#
#         if not hasattr(self.call, 'function_spec') and self.tf_function:
#
#             tf_function_kwargs = {
#                 'input_signature': [args[0].unspecific_spec],
#                 'experimental_relax_shapes': True}
#
#             self.call = tf.function(self.call, **tf_function_kwargs)
#
#         return super().__call__(*args, **kwargs)
#
#     @abstractmethod
#     def call(self, inputs: GraphTensor):
#         pass
