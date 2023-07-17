from tensorflow import keras

from tensorflow.python.framework import type_spec

from keras import layers


try:
    register_keras_serializable = keras.saving.register_keras_serializable
except AttributeError:
    register_keras_serializable = keras.utils.register_keras_serializable

try:
    PreprocessingLayer = layers.PreprocessingLayer
except AttributeError:
    PreprocessingLayer = layers.experimental.preprocessing.PreprocessingLayer

try:
    from tensorflow.python.framework import type_spec_registry
except ImportError:
    type_spec_registry = None

type_spec_registry = (
    type_spec_registry.register if type_spec_registry is not None 
    else type_spec.register
)

try:
    from keras.engine import keras_tensor
except ImportError:
    from keras.src.engine import keras_tensor

try:
    from keras.layers import core as keras_core
except ImportError:
    from keras.src.layers import core as keras_core
