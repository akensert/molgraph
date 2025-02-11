from tensorflow import keras
from tensorflow.keras import layers


try:
    register_keras_serializable = keras.saving.register_keras_serializable
except AttributeError:
    register_keras_serializable = keras.utils.register_keras_serializable

try:
    PreprocessingLayer = layers.PreprocessingLayer
except AttributeError:
    PreprocessingLayer = layers.experimental.preprocessing.PreprocessingLayer
