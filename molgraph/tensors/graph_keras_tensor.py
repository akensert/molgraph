from molgraph.internal import keras_core 
from molgraph.internal import keras_tensor

from molgraph.tensors.graph_tensor import GraphTensor


class GraphKerasTensor(keras_tensor.KerasTensor):

    @property
    def dtype(self):
        return self.spec.dtype

    @property
    def rank(self):
        return self.spec.rank

    @property
    def spec(self):
        return self._type_spec


tensor_graph_operators = [
    '__getitem__', '__getattr__', 
]
for o in tensor_graph_operators:
    GraphKerasTensor._overload_operator(GraphTensor, o)

tensor_graph_properties = [
    '_data', '_spec',
]
for p in tensor_graph_properties:
    keras_core._delegate_property(GraphKerasTensor, p)

tensor_graph_methods = [
    'update', 'remove', 'merge', 'separate',
]
for m in tensor_graph_methods:
    keras_core._delegate_method(GraphKerasTensor, m)


# from tensorflow.python.util import dispatch
#
# class TFClassMethodDispatcher(dispatch.OpDispatcher):
#     """This class is defined as it could not be imported from keras.layers.core;
#     reference:
#         https://github.com/tensorflow/tensorflow/blob/master/
#         tensorflow/python/keras/layers/core.py
#     """
#
#     def __init__(self, cls, method_name):
#         self.cls = cls
#         self.method_name = method_name
#
#     def handle(self, args, kwargs):
#         if any(
#             isinstance(x, keras_tensor.KerasTensor)
#             for x in nest.flatten([args, kwargs])):
#             return core.ClassMethod(self.cls, self.method_name)(args[1:], kwargs)
#         else:
#             return object()
#
# tensor_graph_class_methods = [
#     'class_methods_goes_in_here',
# ]
# for cm in tensor_graph_class_methods:
#     TFClassMethodDispatcher(GraphTensor, cm).register(
#         getattr(GraphTensor, cm))


keras_tensor.register_keras_tensor_specialization(
    GraphTensor, GraphKerasTensor)
