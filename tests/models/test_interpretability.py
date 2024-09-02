import molgraph

import unittest

import tensorflow as tf

import tempfile
import shutil

from molgraph.layers import GTConv, Readout

from molgraph.models import GradientActivationMapping
from molgraph.models import SaliencyMapping
from molgraph.models import IntegratedSaliencyMapping
from molgraph.models import SmoothGradSaliencyMapping

from tests.models._common import graph_tensor_2 as graph_tensor



class TestSaliency(unittest.TestCase):

    def _test_saliency(
        self,
        inputs, 
        label, 
        merge,
        saliency_class, 
        output_units, 
        activation):
        
        if merge:
            inputs = inputs.merge()
            input_spec = inputs.spec
        else:
            input_spec = inputs.spec
        
        sequential_model = tf.keras.Sequential([
            molgraph.layers.GNNInputLayer(type_spec=input_spec),
            GTConv(32, name='conv_1'),
            GTConv(32, name='conv_2'),
            Readout(),
            tf.keras.layers.Dense(64),
            tf.keras.layers.Dense(output_units)
        ])

        saliency_model = saliency_class(
            sequential_model, 
            activation,
            random_seed=42,
        )

        if label is None:
            input_signature = [input_spec]
        else:
            input_signature = [
                input_spec, 
                tf.TensorSpec(tf.TensorShape([None]).concatenate(label.shape[1:]))]

        saliency_model.__call__ = tf.function(
            saliency_model.__call__, input_signature=input_signature
        )

        maps_1 = saliency_model(inputs, label)

        file = tempfile.NamedTemporaryFile()
        filename = file.name
        file.close()


        tf.saved_model.save(saliency_model, filename)
        saliency_model_loaded = tf.saved_model.load(filename)
        shutil.rmtree(filename)

        if label is not None:
            maps_2 = saliency_model_loaded(inputs, label)
        else:
            maps_2 = saliency_model_loaded(inputs)

        if merge:
            test1 = all(
                maps_1.numpy().round(3) ==  maps_2.numpy().round(3))
        else:
            test1 = all(
                maps_1.flat_values.numpy().round(3) == 
                maps_2.flat_values.numpy().round(3))
            
        self.assertTrue(test1)

    def test_vanilla_saliency(self):
        self._test_saliency(graph_tensor, None, True, SaliencyMapping, 1, 'linear')
        self._test_saliency(graph_tensor, None, False, SaliencyMapping, 1, 'linear')

    def test_integrated_saliency(self):
        self._test_saliency(graph_tensor, None, True, IntegratedSaliencyMapping, 1, 'linear')
        self._test_saliency(graph_tensor, None, False, IntegratedSaliencyMapping, 1, 'linear')

    def test_smoothgrad_saliency(self):
        self._test_saliency(graph_tensor, None, True, SmoothGradSaliencyMapping, 1, 'linear')
        self._test_saliency(graph_tensor, None, False, SmoothGradSaliencyMapping, 1, 'linear')

    def test_vanilla_saliency_with_label(self):
        label = tf.constant([1., 2., 3., 4., 5.])
        self._test_saliency(graph_tensor, label, True, SaliencyMapping, 1, 'linear')
        self._test_saliency(graph_tensor, label, False, SaliencyMapping, 1, 'linear')

    def test_integrated_saliency_with_label(self):
        label = tf.constant([1., 2., 3., 4., 5.])
        self._test_saliency(graph_tensor, label, True, IntegratedSaliencyMapping, 1, 'linear')
        self._test_saliency(graph_tensor, label, False, IntegratedSaliencyMapping, 1, 'linear')

    def test_smoothgrad_saliency_with_label(self):
        label = tf.constant([1., 2., 3., 4., 5.])
        self._test_saliency(graph_tensor, label, True, SmoothGradSaliencyMapping, 1, 'linear')
        self._test_saliency(graph_tensor, label, False, SmoothGradSaliencyMapping, 1, 'linear')

    def test_vanilla_saliency_with_onehot_label(self):
        label = tf.constant([
            [0., 1., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 0., 0.],
            [0., 1., 0.],
        ])
        self._test_saliency(graph_tensor, label, True, SaliencyMapping, 3, 'softmax')
        self._test_saliency(graph_tensor, label, False, SaliencyMapping, 3, 'softmax')

    def test_integrated_saliency_with_onehot_label(self):
        label = tf.constant([
            [0., 1., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 0., 0.],
            [0., 1., 0.],
        ])
        self._test_saliency(graph_tensor, label, True, IntegratedSaliencyMapping, 3, 'softmax')
        self._test_saliency(graph_tensor, label, False, IntegratedSaliencyMapping, 3, 'softmax')

    def test_smoothgrad_saliency_with_onehot_label(self):
        label = tf.constant([
            [0., 1., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 0., 0.],
            [0., 1., 0.],
        ])
        self._test_saliency(graph_tensor, label, True, SmoothGradSaliencyMapping, 3, 'softmax')
        self._test_saliency(graph_tensor, label, False, SmoothGradSaliencyMapping, 3, 'softmax')


class TestGradientActivation(unittest.TestCase):

    def _test_gradient_activation(
        self,
        inputs, 
        label, 
        merge,
        output_units, 
        activation,
    ):
        
        if merge:
            inputs = inputs.merge()
            input_spec = inputs.spec
        else:
            input_spec = inputs.spec

        sequential_model = tf.keras.Sequential([
            molgraph.layers.GNNInputLayer(type_spec=input_spec),
            GTConv(32, name='conv_1'),
            GTConv(32, name='conv_2'),
            Readout(),
            tf.keras.layers.Dense(64),
            tf.keras.layers.Dense(output_units)
        ])

        saliency_model = GradientActivationMapping(
            sequential_model, 
            layer_names=['conv_1', 'conv_2'],
            output_activation=activation,
            random_seed=42,
        )

        if label is None:
            input_signature = [input_spec]
        else:
            input_signature = [
                input_spec, 
                tf.TensorSpec(tf.TensorShape([None]).concatenate(label.shape[1:]))]

        saliency_model.__call__ = tf.function(
            saliency_model.__call__, input_signature=input_signature
        )

        maps_1 = saliency_model(inputs, label)

        file = tempfile.NamedTemporaryFile()
        filename = file.name
        file.close()
        tf.saved_model.save(saliency_model, filename)
        saliency_model_loaded = tf.saved_model.load(filename)
        shutil.rmtree(filename)

        if label is not None:
            maps_2 = saliency_model_loaded(inputs, label)
        else:
            maps_2 = saliency_model_loaded(inputs)
        
        if merge:
            test1 = all(
                maps_1.numpy().round(3) ==  maps_2.numpy().round(3))
        else:
            test1 = all(
                maps_1.flat_values.numpy().round(3) == 
                maps_2.flat_values.numpy().round(3))
            
        self.assertTrue(test1)

    def test_gradient_activation(self):
        self._test_gradient_activation(graph_tensor, None, True, 1, 'linear')
        self._test_gradient_activation(graph_tensor, None, False, 1, 'linear')

    def test_gradient_activation_with_label(self):
        label = tf.constant([1., 2., 3., 4., 5.])
        self._test_gradient_activation(graph_tensor, label, True, 1, 'linear')
        self._test_gradient_activation(graph_tensor, label, False, 1, 'linear')
    
    def test_gradient_activation_with_onehot_label(self):
        label = tf.constant([
            [0., 1., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 0., 0.],
            [0., 1., 0.],
        ])
        self._test_gradient_activation(graph_tensor, label, True, 3, 'softmax')
        self._test_gradient_activation(graph_tensor, label, False, 3, 'softmax')

    def _test_gradient_activation_without_layer_names(
        self,
        inputs, 
        label, 
        merge,
        output_units, 
        activation,
    ):
        
        if merge:
            inputs = inputs.merge()
            input_spec = inputs.spec
        else:
            input_spec = inputs.spec

        sequential_model = tf.keras.Sequential([
            molgraph.layers.GNNInputLayer(type_spec=input_spec),
            GTConv(32, name='conv_1'),
            GTConv(32, name='conv_2'),
            Readout(),
            tf.keras.layers.Dense(64),
            tf.keras.layers.Dense(output_units)
        ])

        saliency_model = GradientActivationMapping(
            sequential_model, 
            output_activation=activation,
            random_seed=42,
        )

        if label is None:
            input_signature = [input_spec]
        else:
            input_signature = [
                input_spec, 
                tf.TensorSpec(tf.TensorShape([None]).concatenate(label.shape[1:]))]

        saliency_model.__call__ = tf.function(
            saliency_model.__call__, input_signature=input_signature
        )

        maps_1 = saliency_model(inputs, label)

        file = tempfile.NamedTemporaryFile()
        filename = file.name
        file.close()
        tf.saved_model.save(saliency_model, filename)
        saliency_model_loaded = tf.saved_model.load(filename)
        shutil.rmtree(filename)

        if label is not None:
            maps_2 = saliency_model_loaded(inputs, label)
        else:
            maps_2 = saliency_model_loaded(inputs)
        
        if merge:
            test1 = all(
                maps_1.numpy().round(3) ==  maps_2.numpy().round(3))
        else:
            test1 = all(
                maps_1.flat_values.numpy().round(3) == 
                maps_2.flat_values.numpy().round(3))
            
        self.assertTrue(test1)

    def test_gradient_activation_without_layer_names(self):
        self._test_gradient_activation_without_layer_names(graph_tensor, None, True, 1, 'linear')
        self._test_gradient_activation_without_layer_names(graph_tensor, None, False, 1, 'linear')

    def test_gradient_activation_with_label_and_without_layer_names(self):
        label = tf.constant([1., 2., 3., 4., 5.])
        self._test_gradient_activation_without_layer_names(graph_tensor, label, True, 1, 'linear')
        self._test_gradient_activation_without_layer_names(graph_tensor, label, False, 1, 'linear')
    
    def test_gradient_activation_with_onehot_label_and_without_layer_names(self):
        label = tf.constant([
            [0., 1., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 0., 0.],
            [0., 1., 0.],
        ])
        self._test_gradient_activation_without_layer_names(graph_tensor, label, True, 3, 'softmax')
        self._test_gradient_activation_without_layer_names(graph_tensor, label, False, 3, 'softmax')

    def _test_gradient_activation_with_gnn(
        self,
        inputs, 
        label, 
        merge,
        output_units, 
        activation,
    ):
        
        if merge:
            inputs = inputs.merge()
            input_spec = inputs.spec
        else:
            input_spec = inputs.spec

        sequential_model = tf.keras.Sequential([
            molgraph.layers.GNNInputLayer(type_spec=input_spec),
            molgraph.layers.GNN([
                molgraph.layers.FeatureProjection(32, name='proj'),
                GTConv(32, name='conv_1'),
                GTConv(32, name='conv_2'),
                GTConv(32, name='conv_3'),
            ]),
            Readout(),
            tf.keras.layers.Dense(64),
            tf.keras.layers.Dense(output_units)
        ])

        saliency_model = GradientActivationMapping(
            sequential_model, 
            output_activation=activation,
            random_seed=42,
        )

        if label is None:
            input_signature = [input_spec]
        else:
            input_signature = [
                input_spec, 
                tf.TensorSpec(tf.TensorShape([None]).concatenate(label.shape[1:]))]

        saliency_model.__call__ = tf.function(
            saliency_model.__call__, input_signature=input_signature
        )

        maps_1 = saliency_model(inputs, label)

        file = tempfile.NamedTemporaryFile()
        filename = file.name
        file.close()
        tf.saved_model.save(saliency_model, filename)
        saliency_model_loaded = tf.saved_model.load(filename)
        shutil.rmtree(filename)

        if label is not None:
            maps_2 = saliency_model_loaded(inputs, label)
        else:
            maps_2 = saliency_model_loaded(inputs)
        
        if merge:
            test1 = all(
                maps_1.numpy().round(3) ==  maps_2.numpy().round(3))
        else:
            test1 = all(
                maps_1.flat_values.numpy().round(3) == 
                maps_2.flat_values.numpy().round(3))
            
        self.assertTrue(test1)

    def test_gradient_activation_with_gnn(self):
        self._test_gradient_activation_with_gnn(graph_tensor, None, True, 1, 'linear')
        self._test_gradient_activation_with_gnn(graph_tensor, None, False, 1, 'linear')

    def test_gradient_activation_with_label_and_with_gnn(self):
        label = tf.constant([1., 2., 3., 4., 5.])
        self._test_gradient_activation_with_gnn(graph_tensor, label, True, 1, 'linear')
        self._test_gradient_activation_with_gnn(graph_tensor, label, False, 1, 'linear')
    
    def test_gradient_activation_with_onehot_label_and_with_gnn(self):
        label = tf.constant([
            [0., 1., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 0., 0.],
            [0., 1., 0.],
        ])
        self._test_gradient_activation_with_gnn(graph_tensor, label, True, 3, 'softmax')
        self._test_gradient_activation_with_gnn(graph_tensor, label, False, 3, 'softmax')

if __name__ == "__main__":
    unittest.main()
