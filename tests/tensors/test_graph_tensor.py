from tests.tensors._common import graph_tensor_1
from tests.tensors._common import graph_tensor_1_12
from tests.tensors._common import graph_tensor_2
from tests.tensors._common import graph_tensor_3
from tests.tensors._common import graph_tensor_123
from tests.tensors._common import graph_tensor_4
from tests.tensors._common import graph_tensor_5

import tensorflow as tf
import unittest

from molgraph.tensors.graph_tensor import GraphTensor


inputs = [graph_tensor_1, graph_tensor_2, graph_tensor_3]


class TestGraphTensor(unittest.TestCase):

    def _test_shape(
        self, 
        graph_tensor: GraphTensor,
        num_graphs: int,
        num_nodes: int,
        num_edges: int,
    ):
        if isinstance(graph_tensor.node_feature, tf.RaggedTensor):
            self.assertSequenceEqual(
                graph_tensor.node_feature.shape.as_list(), 
                [num_graphs, None, 11])
            self.assertSequenceEqual(
                graph_tensor.edge_feature.shape.as_list(), 
                [num_graphs, None, 5])
            self.assertEqual(
                sum(graph_tensor.node_feature.row_lengths()), num_nodes)
            self.assertEqual(
                sum(graph_tensor.edge_feature.row_lengths()), num_edges)
            self.assertEqual(
                sum(graph_tensor.edge_src.row_lengths()), num_edges)
            self.assertEqual(
                sum(graph_tensor.edge_dst.row_lengths()), num_edges)
        else:
            self.assertSequenceEqual(
                graph_tensor.node_feature.shape.as_list(), [num_nodes, 11])
            self.assertSequenceEqual(
                graph_tensor.edge_feature.shape.as_list(), [num_edges, 5])
            self.assertSequenceEqual(
                graph_tensor.edge_src.shape.as_list(), [num_edges])
            self.assertSequenceEqual(
                graph_tensor.edge_dst.shape.as_list(), [num_edges])
            self.assertSequenceEqual(
                graph_tensor.graph_indicator.shape.as_list(), [num_nodes])
            self.assertEqual(
                tf.unique(graph_tensor.graph_indicator)[0].shape.as_list()[0], 
                num_graphs)
            
    def _test_dtype(self, graph_tensor, edge_dtype=tf.int32):
        self.assertIs(graph_tensor.node_feature.dtype, tf.float32)
        self.assertIs(graph_tensor.edge_feature.dtype, tf.float32)
        self.assertIs(graph_tensor.edge_src.dtype, edge_dtype)
        self.assertIs(graph_tensor.edge_dst.dtype, edge_dtype)
        if graph_tensor.graph_indicator is not None:
            self.assertIs(graph_tensor.graph_indicator.dtype, edge_dtype)

    def test_merge(self):
        merged_graph_tensor_1 = graph_tensor_1.merge()
        self._test_shape(merged_graph_tensor_1, 3, 32, 64)
        self._test_dtype(merged_graph_tensor_1)

        merged_graph_tensor_2 = graph_tensor_2.merge()
        self._test_shape(merged_graph_tensor_2, 1, 1, 0)
        self._test_dtype(merged_graph_tensor_2)

        merged_graph_tensor_3 = graph_tensor_3.merge()
        self._test_shape(merged_graph_tensor_3, 3, 5, 4)
        self._test_dtype(merged_graph_tensor_3)

        merged_graph_tensor_4 = graph_tensor_4.merge()
        self._test_shape(merged_graph_tensor_4, 3, 32, 64)
        self._test_dtype(merged_graph_tensor_4, tf.int64)

    def test_separate(self):

        separated_graph_tensor_1 = graph_tensor_1.merge().separate()
        self._test_shape(separated_graph_tensor_1, 3, 32, 64)
        self._test_dtype(separated_graph_tensor_1)

        separated_graph_tensor_2 = graph_tensor_2.merge().separate()
        self._test_shape(separated_graph_tensor_2, 1, 1, 0)
        self._test_dtype(separated_graph_tensor_2)

        separated_graph_tensor_3 = graph_tensor_3.merge().separate()
        self._test_shape(separated_graph_tensor_3, 3, 5, 4)
        self._test_dtype(separated_graph_tensor_3)

        separated_graph_tensor_4 = graph_tensor_4.merge().separate()
        self._test_shape(separated_graph_tensor_4, 3, 32, 64)
        self._test_dtype(separated_graph_tensor_4, tf.int64)

    def test_merge_propagate(self):
        merged_graph_tensor_1 = graph_tensor_1.merge()
        merged_graph_tensor_1 = merged_graph_tensor_1.propagate()
        self._test_shape(merged_graph_tensor_1, 3, 32, 64)
        self._test_dtype(merged_graph_tensor_1)

        merged_graph_tensor_2 = graph_tensor_2.merge()
        merged_graph_tensor_2 = merged_graph_tensor_2.propagate()
        self._test_shape(merged_graph_tensor_2, 1, 1, 0)
        self._test_dtype(merged_graph_tensor_2)

        merged_graph_tensor_3 = graph_tensor_3.merge()
        merged_graph_tensor_3 = merged_graph_tensor_3.propagate()
        self._test_shape(merged_graph_tensor_3, 3, 5, 4)
        self._test_dtype(merged_graph_tensor_3)

        merged_graph_tensor_4 = graph_tensor_4.merge()
        merged_graph_tensor_4 = merged_graph_tensor_4.propagate()
        self._test_shape(merged_graph_tensor_4, 3, 32, 64)
        self._test_dtype(merged_graph_tensor_4, tf.int64)

    def test_separate_propagate(self):
        separated_graph_tensor_1 = graph_tensor_1.merge().separate()
        separated_graph_tensor_1 = separated_graph_tensor_1.propagate()
        self._test_shape(separated_graph_tensor_1, 3, 32, 64)
        self._test_dtype(separated_graph_tensor_1)

        separated_graph_tensor_2 = graph_tensor_2.merge().separate()
        separated_graph_tensor_2 = separated_graph_tensor_2.propagate()
        self._test_shape(separated_graph_tensor_2, 1, 1, 0)
        self._test_dtype(separated_graph_tensor_2)

        separated_graph_tensor_3 = graph_tensor_3.merge().separate()
        separated_graph_tensor_3 = separated_graph_tensor_3.propagate()
        self._test_shape(separated_graph_tensor_3, 3, 5, 4)
        self._test_dtype(separated_graph_tensor_3)

        separated_graph_tensor_4 = graph_tensor_4.merge().separate()
        separated_graph_tensor_4 = separated_graph_tensor_4.propagate()
        self._test_shape(separated_graph_tensor_4, 3, 32, 64)
        self._test_dtype(separated_graph_tensor_4, tf.int64)

    def test_merge_separate_merge_separate(self):

        for t in inputs:
            t = t.update({'node_feature_2': t.node_feature[:, :, :-1],
                          'edge_feature_2': t.edge_feature[:, :, :-1]})
            t_2 = t.merge().separate().merge().separate()
            test_1 = tf.shape(t_2.node_feature) == tf.shape(t.node_feature)
            test_2 = tf.shape(t_2.edge_feature) == tf.shape(t.edge_feature)
            test_3 = tf.shape(t_2.node_feature_2) == tf.shape(t.node_feature_2)
            test_4 = tf.shape(t_2.edge_feature_2) == tf.shape(t.edge_feature_2)
            test_5 = tf.shape(t_2.edge_src) == tf.shape(t.edge_src)
            test_6 = tf.shape(t_2.edge_dst) == tf.shape(t.edge_dst)
            test = tf.reduce_all(tf.stack([
                test_1, test_2, test_3, test_4, test_5, test_6], axis=0))
            self.assertTrue(test.numpy())

    def test_separate_merge_separate_merge(self):

        for t in inputs:
            t = t.merge()
            t = t.update({'node_feature_2': t.node_feature[:, :-1],
                          'edge_feature_2': t.edge_feature[:, :-1]})
            t_2 = t.separate().merge().separate().merge()
            test_1 = tf.shape(t_2.node_feature) == tf.shape(t.node_feature)
            test_2 = tf.shape(t_2.edge_feature) == tf.shape(t.edge_feature)
            test_3 = tf.shape(t_2.node_feature_2) == tf.shape(t.node_feature_2)
            test_4 = tf.shape(t_2.edge_feature_2) == tf.shape(t.edge_feature_2)
            test_5 = tf.shape(t_2.edge_src) == tf.shape(t.edge_src)
            test_6 = tf.shape(t_2.edge_dst) == tf.shape(t.edge_dst)
            test = tf.reduce_all(tf.concat([
                test_1, test_2, test_3, test_4, test_5, test_6], axis=0))
            self.assertTrue(test.numpy())

    def test_nonragged_to_nonragged_update(self):
        for t in inputs:
            t = t.merge()
            t = t.update({
                'node_feature_1': t.node_feature,
                'edge_feature_1': t.edge_feature,
            })
            test1 = tf.reduce_all(
                tf.shape(t.node_feature) == tf.shape(t.node_feature_1)
            ).numpy()
            test2 = tf.reduce_all(
                tf.shape(t.edge_feature) == tf.shape(t.edge_feature_1)
            ).numpy()
            self.assertTrue(test1)
            self.assertTrue(test2)

            t_2 = t.update({
                'node_feature': t.node_feature,
                'edge_feature': t.edge_feature,
            })
            test1 = tf.reduce_all(
                tf.shape(t.node_feature) == tf.shape(t_2.node_feature)
            ).numpy()
            test2 = tf.reduce_all(
                tf.shape(t.edge_feature) == tf.shape(t_2.edge_feature)
            ).numpy()
            self.assertTrue(test1)
            self.assertTrue(test2)

    def test_ragged_to_nonragged_update(self):
        for t in inputs:
            t_merged = t.merge()
            t_merged = t_merged.update({
                'node_feature_1': t.node_feature,
                'edge_feature_1': t.edge_feature,
            })
            t = t_merged
            test1 = tf.reduce_all(
                tf.shape(t.node_feature) == tf.shape(t.node_feature_1)
            ).numpy()
            test2 = tf.reduce_all(
                tf.shape(t.edge_feature) == tf.shape(t.edge_feature_1)
            ).numpy()
            self.assertTrue(test1)
            self.assertTrue(test2)

            t_merged_2 = t_merged.update({
                'node_feature': t_merged.node_feature,
                'edge_feature': t_merged.edge_feature,
            })
            test1 = tf.reduce_all(
                tf.shape(t.node_feature) == tf.shape(t_merged_2.node_feature)
            ).numpy()
            test2 = tf.reduce_all(
                tf.shape(t.edge_feature) == tf.shape(t_merged_2.edge_feature)
            ).numpy()
            self.assertTrue(test1)
            self.assertTrue(test2)

    def test_different_dim_nonragged_to_nonragged_update(self):
        for t in inputs:
            t = t.merge()
            t = t.update({
                'node_feature_1': t.node_feature[:, :-1],
                'edge_feature_1': t.edge_feature[:, :-1],
            })
            test1 = tf.reduce_all(
                tf.shape(t.node_feature[:, :-1]) == tf.shape(t.node_feature_1)
            ).numpy()
            test2 = tf.reduce_all(
                tf.shape(t.edge_feature[:, :-1]) == tf.shape(t.edge_feature_1)
            ).numpy()
            self.assertTrue(test1)
            self.assertTrue(test2)

            t_2 = t.update({
                'node_feature': t.node_feature[:, :-1],
                'edge_feature': t.edge_feature[:, :-1],
            })
            test1 = tf.reduce_all(
                tf.shape(t.node_feature[:, :-1]) == tf.shape(t_2.node_feature)
            ).numpy()
            test2 = tf.reduce_all(
                tf.shape(t.edge_feature[:, :-1]) == tf.shape(t_2.edge_feature)
            ).numpy()
            self.assertTrue(test1)
            self.assertTrue(test2)

    def test_different_dim_ragged_to_ragged_update(self):
        for t in inputs:
            t = t.update({
                'node_feature_1': t.node_feature[:, :, :-1],
                'edge_feature_1': t.edge_feature[:, :, :-1],
            })
            test1 = tf.shape(t.node_feature[:, :, :-1]) == tf.shape(t.node_feature_1)
            test2 = tf.shape(t.edge_feature[:, :, :-1]) == tf.shape(t.edge_feature_1)
            self.assertTrue(test1.numpy())
            self.assertTrue(test2.numpy())

            t_2 = t.update({
                'node_feature': t.node_feature[:, :, :-1],
                'edge_feature': t.edge_feature[:, :, :-1],
            })
            test1 = tf.shape(t.node_feature[:, :, :-1]) == tf.shape(t_2.node_feature)
            test2 = tf.shape(t.edge_feature[:, :, :-1]) == tf.shape(t_2.edge_feature)
            self.assertTrue(test1.numpy())
            self.assertTrue(test2.numpy())


    def test_different_dim_nonragged_to_ragged_update(self):
        for t in inputs:
            t_merged = t.merge()
            t = t.update({
                'node_feature_1': t_merged.node_feature[:, :-1],
                'edge_feature_1': t_merged.edge_feature[:, :-1],
            })
            test1 = tf.shape(t.node_feature[:, :, :-1]) == tf.shape(t.node_feature_1)
            test2 = tf.shape(t.edge_feature[:, :, :-1]) == tf.shape(t.edge_feature_1)
            self.assertTrue(test1.numpy())
            self.assertTrue(test2.numpy())

            t_2 = t.update({
                'node_feature': t_merged.node_feature[:, :-1],
                'edge_feature': t_merged.edge_feature[:, :-1],
            })
            test1 = tf.shape(t.node_feature[:, :, :-1]) == tf.shape(t_2.node_feature)
            test2 = tf.shape(t.edge_feature[:, :, :-1]) == tf.shape(t_2.edge_feature)
            self.assertTrue(test1.numpy())
            self.assertTrue(test2.numpy())

    def test_different_dim_ragged_to_nonragged_update(self):
        for t in inputs:
            t_merged = t.merge()
            t_merged = t_merged.update({
                'node_feature_1': t.node_feature[:, :, :-1],
                'edge_feature_1': t.edge_feature[:, :, :-1],
            })
            t = t_merged
            test1 = tf.reduce_all(
                tf.shape(t.node_feature[:, :-1]) == tf.shape(t.node_feature_1)
            ).numpy()
            test2 = tf.reduce_all(
                tf.shape(t.edge_feature[:, :-1]) == tf.shape(t.edge_feature_1)
            ).numpy()
            self.assertTrue(test1)
            self.assertTrue(test2)

            t_merged_2 = t_merged.update({
                'node_feature': t_merged.node_feature[:, :-1],
                'edge_feature': t_merged.edge_feature[:, :-1],
            })
            test1 = tf.reduce_all(
                tf.shape(t.node_feature[:, :-1]) == tf.shape(t_merged_2.node_feature)
            ).numpy()
            test2 = tf.reduce_all(
                tf.shape(t.edge_feature[:, :-1]) == tf.shape(t_merged_2.edge_feature)
            ).numpy()
            self.assertTrue(test1)
            self.assertTrue(test2)

    def test_node_size_incompatible_nonragged_to_nonragged_update(self):
        for t in inputs:
            try:
                t = t.merge()
                t.update({
                    'node_feature_1': t.node_feature[:-1],
                })
                exception = None
            except Exception as e:
                exception = e
            self.assertIsInstance(exception, tf.errors.InvalidArgumentError)

            try:
                t.update({
                    'node_feature': t.node_feature[:-1],
                })
                exception = None
            except Exception as e:
                exception = e
            self.assertIsInstance(exception, tf.errors.InvalidArgumentError)

    def test_edge_size_incompatible_nonragged_to_nonragged_update(self):
        for t in inputs:
            if not t.merge().edge_feature.shape[0]:
                continue
            try:
                t = t.merge()
                t.update({
                    'edge_feature_1': t.edge_feature[:-1],
                })
                exception = None
            except Exception as e:
                exception = e
            self.assertIsInstance(exception, tf.errors.InvalidArgumentError)

            try:
                t.update({
                    'edge_feature': t.edge_feature[:-1],
                })
                exception = None
            except Exception as e:
                exception = e
            self.assertIsInstance(exception, tf.errors.InvalidArgumentError)


    def test_node_size_incompatible_ragged_to_nonragged_update(self):
        for t in inputs:
            try:
                t_merged = t.merge()
                t_merged.update({
                    'node_feature_1': t.node_feature[:, :-1, :],
                })
                exception = None
            except Exception as e:
                exception = e
            self.assertIsInstance(exception, tf.errors.InvalidArgumentError)

            try:
                t_merged = t.merge()
                t_merged.update({
                    'node_feature': t.node_feature[:, :-1, :],
                })
                exception = None
            except Exception as e:
                exception = e
            self.assertIsInstance(exception, tf.errors.InvalidArgumentError)
        
    def test_edge_size_incompatible_ragged_to_nonragged_update(self):
        for t in inputs:
            if not t.merge().edge_feature.shape[0]:
                continue
            try:
                t_merged = t.merge()
                t_merged.update({
                    'edge_feature_1': t.edge_feature[:, :-1, :],
                })
                exception = None
            except Exception as e:
                exception = e
            self.assertIsInstance(exception, tf.errors.InvalidArgumentError)

            try:
                t_merged = t.merge()
                t_merged.update({
                    'edge_feature': t.edge_feature[:, :-1, :],
                })
                exception = None
            except Exception as e:
                exception = e
            self.assertIsInstance(exception, tf.errors.InvalidArgumentError)

    def test_ragged_to_ragged_update(self):
        for t in inputs:
            t = t.update({
                'node_feature_1': t.node_feature,
                'edge_feature_1': t.edge_feature,
            })
            test1 = tf.shape(t.node_feature) == tf.shape(t.node_feature_1)
            test2 = tf.shape(t.edge_feature) == tf.shape(t.edge_feature_1)
            self.assertTrue(test1.numpy())
            self.assertTrue(test2.numpy())

            t_2 = t.update({
                'node_feature': t.node_feature,
                'edge_feature': t.edge_feature,
            })
            test1 = tf.shape(t.node_feature) == tf.shape(t_2.node_feature)
            test2 = tf.shape(t.edge_feature) == tf.shape(t_2.edge_feature)
            self.assertTrue(test1.numpy())
            self.assertTrue(test2.numpy())

    def test_nonragged_to_ragged_update(self):
        for t in inputs:
            t_merged = t.merge()
            t = t.update({
                'node_feature_1': t_merged.node_feature,
                'edge_feature_1': t_merged.edge_feature,
            })
            test1 = tf.shape(t.node_feature) == tf.shape(t.node_feature_1)
            test2 = tf.shape(t.edge_feature) == tf.shape(t.edge_feature_1)
            self.assertTrue(test1.numpy())
            self.assertTrue(test2.numpy())

            t_2 = t.update({
                'node_feature_1': t_merged.node_feature,
                'edge_feature_1': t_merged.edge_feature,
            })
            test1 = tf.shape(t.node_feature) == tf.shape(t_2.node_feature)
            test2 = tf.shape(t.edge_feature) == tf.shape(t_2.edge_feature)
            self.assertTrue(test1.numpy())
            self.assertTrue(test2.numpy())

    def test_node_size_incompatible_ragged_to_ragged_update(self):
        for t in inputs:
            try:
                t.update({
                    'node_feature_1': t.node_feature[:, :-1, :],
                })
                exception = None
            except Exception as e:
                exception = e
            self.assertIsInstance(exception, tf.errors.InvalidArgumentError)

            try:
                t.update({
                    'node_feature': t.node_feature[:, :-1, :],
                })
                exception = None
            except Exception as e:
                exception = e
            self.assertIsInstance(exception, tf.errors.InvalidArgumentError)

    def test_edge_size_incompatible_ragged_to_ragged_update(self):
        for t in inputs:
            if not t.merge().edge_feature.shape[0]:
                continue
            try:
                t.update({
                    'edge_feature_1': t.edge_feature[:, :-1, :],
                })
                exception = None
            except Exception as e:
                exception = e
            self.assertIsInstance(exception, tf.errors.InvalidArgumentError)

            try:
                t.update({
                    'edge_feature': t.edge_feature[:, :-1, :],
                })
                exception = None
            except Exception as e:
                exception = e
            self.assertIsInstance(exception, tf.errors.InvalidArgumentError)

    def test_node_size_incompatible_nonragged_to_ragged_update(self):
        for t in inputs:
            try:
                t_merged = t.merge()
                t.update({
                    'node_feature_1': t_merged.node_feature[:-1],
                })
                exception = None
            except Exception as e:
                exception = e
            self.assertIsInstance(exception, ValueError)

            try:
                t_merged = t.merge()
                t.update({
                    'node_feature': t_merged.node_feature[:-1],
                })
                exception = None
            except Exception as e:
                exception = e
            self.assertIsInstance(exception, ValueError)

    def test_edge_size_incompatible_nonragged_to_ragged_update(self):
        for t in inputs:
            if not t.merge().edge_feature.shape[0]:
                continue
            try:
                t_merged = t.merge()
                t.update({
                    'edge_feature_1': t_merged.edge_feature[:-1],
                })
                exception = None
            except Exception as e:
                exception = e
            self.assertIsInstance(exception, ValueError)

            try:
                t_merged = t.merge()
                t.update({
                    'edge_feature': t_merged.edge_feature[:-1],
                })
                exception = None
            except Exception as e:
                exception = e
            self.assertIsInstance(exception, ValueError)

    def test_remove(self):
        for t in inputs:
            t = t.update({'edge_feature_2': t.edge_feature})
            t = t.remove(['edge_feature', 'edge_feature_2'])
            test = t.edge_feature is None and not hasattr(t, 'edge_feature_2')
            self.assertTrue(test)
            try:
                t.remove(['edge_feature_3'])
                exception = None
            except Exception as e:
                exception = e
            self.assertIsInstance(exception, KeyError)

            try:
                t.remove(['edge_feature'])
                exception = None
            except Exception as e:
                exception = e
            self.assertIsInstance(exception, KeyError)

    def test_random_edge_order_propagate(self):
        for t in inputs:
            t = t.merge()
            t = t.update({
                'edge_weight': tf.random.uniform(
                    shape=(tf.shape(t.edge_src)[0], 1)
                )
            })
            edge_src = t.edge_src
            edge_dst = t.edge_dst     
            random_indices = tf.random.shuffle(
                tf.range(tf.shape(edge_src)[0]))
            t_random = t.__class__(
                node_feature=t.node_feature,
                edge_src=tf.gather(edge_src, random_indices),
                edge_dst=tf.gather(edge_dst, random_indices),
                edge_feature=tf.gather(t.edge_feature, random_indices),
                edge_weight=tf.gather(t.edge_weight, random_indices),
            )

            test_1 = tf.reduce_all(tf.equal(
                tf.math.round(t.propagate().node_feature, 3),
                tf.math.round(t_random.propagate().node_feature, 3)
            )).numpy()


            self.assertTrue(test_1)

            t = t.separate()
            t_random = t_random.separate()
            
            test_2 = tf.reduce_all(tf.equal(
                tf.math.round(t.propagate().merge().node_feature, 3),
                tf.math.round(t_random.propagate().merge().node_feature, 3)
            )).numpy()

            self.assertTrue(test_2)

    def test_normalize_propagate(self):

        for t in inputs:
            t = t.merge()
            t = t.update({
                'edge_weight': tf.random.uniform(
                    shape=(tf.shape(t.edge_src)[0], 1)
                )
            })
            edge_src = t.edge_src
            edge_dst = t.edge_dst     
            random_indices = tf.random.shuffle(
                tf.range(tf.shape(edge_src)[0]))
            t_random = t.__class__(
                node_feature=t.node_feature,
                edge_src=tf.gather(edge_src, random_indices),
                edge_dst=tf.gather(edge_dst, random_indices),
                edge_feature=tf.gather(t.edge_feature, random_indices),
                edge_weight=tf.gather(t.edge_weight, random_indices),
            )

            test_1 = tf.reduce_all(tf.equal(
                tf.math.round(t.propagate(
                    mode='mean', 
                    normalize=True).node_feature, 3),
                tf.math.round(t_random.propagate(
                    mode='mean', 
                    normalize=True).node_feature, 3)
            )).numpy()


            self.assertTrue(test_1)

            t = t.separate()
            t_random = t_random.separate()
            
            test_2 = tf.reduce_all(tf.equal(
                tf.math.round(t.propagate(
                    mode='mean', 
                    normalize=True).merge().node_feature, 3),
                tf.math.round(t_random.propagate(
                    mode='mean', 
                    normalize=True).merge().node_feature, 3)
            )).numpy()
            
            self.assertTrue(test_2)

    def test_nonragged_normalize_propagate_residual_reduce(self):
        for t in inputs:
            t = t.merge()
            t = t.update({
                'node_feature': tf.stack([
                    t.node_feature, 
                    t.node_feature + tf.random.uniform(t.node_feature.shape),
                    t.node_feature + tf.random.uniform(t.node_feature.shape),
                ], axis=1),
                'edge_weight': tf.random.uniform(
                    shape=(tf.shape(t.edge_src)[0], 3, 1)
                )
            })
            edge_src = t.edge_src
            edge_dst = t.edge_dst     
            random_indices = tf.random.shuffle(
                tf.range(tf.shape(edge_src)[0]))
            t_random = t.__class__(
                node_feature=t.node_feature,
                edge_src=tf.gather(edge_src, random_indices),
                edge_dst=tf.gather(edge_dst, random_indices),
                edge_feature=tf.gather(t.edge_feature, random_indices),
                edge_weight=tf.gather(t.edge_weight, random_indices),
            )

            residual = tf.random.uniform(t.node_feature.shape)

            test = tf.reduce_all(tf.equal(
                tf.math.round(t.propagate(
                    mode='mean', 
                    normalize=True, 
                    residual=residual, 
                    reduce='concat').node_feature, 3),
                tf.math.round(t_random.propagate(
                    mode='mean', 
                    normalize=True, 
                    residual=residual, 
                    reduce='concat').node_feature, 3)
            )).numpy()

            self.assertTrue(test)

    def test_ragged_normalize_propagate_residual_reduce(self):
        for t in inputs:
            t = t.merge()
            t = t.update({
                'node_feature': tf.stack([
                    t.node_feature, 
                    t.node_feature + tf.random.uniform(t.node_feature.shape),
                    t.node_feature + tf.random.uniform(t.node_feature.shape),
                ], axis=1),
                'edge_weight': tf.random.uniform(
                    shape=(tf.shape(t.edge_src)[0], 3, 1)
                )
            })
            edge_src = t.edge_src
            edge_dst = t.edge_dst     
            random_indices = tf.random.shuffle(
                tf.range(tf.shape(edge_src)[0]))
            t_random = t.__class__(
                node_feature=t.node_feature,
                edge_src=tf.gather(edge_src, random_indices),
                edge_dst=tf.gather(edge_dst, random_indices),
                edge_feature=tf.gather(t.edge_feature, random_indices),
                edge_weight=tf.gather(t.edge_weight, random_indices),
            )
            
            random_feature = tf.random.uniform(t.node_feature.shape)
            t = t.update({'node_feature_residual': random_feature})
            t_random = t_random.update({'node_feature_residual': random_feature})
            
            t = t.separate()
            t_random = t_random.separate()
            
            test = tf.reduce_all(tf.equal(
                tf.math.round(t.propagate(
                    mode='mean', 
                    normalize=True, 
                    residual=t.node_feature_residual, 
                    reduce='concat').merge().node_feature, 3),
                tf.math.round(t_random.propagate(
                    mode='mean', 
                    normalize=True, 
                    residual=t_random.node_feature_residual, 
                    reduce='concat').merge().node_feature, 3)
            )).numpy()
            
            self.assertTrue(test)

    def test_nonragged_normalize_propagate_ragged_residual_reduce(self):
        for t in inputs:
            t = t.merge()
            t = t.update({
                'node_feature': tf.stack([
                    t.node_feature, 
                    t.node_feature + tf.random.uniform(t.node_feature.shape),
                    t.node_feature + tf.random.uniform(t.node_feature.shape),
                ], axis=1),
                'edge_weight': tf.random.uniform(
                    shape=(tf.shape(t.edge_src)[0], 3, 1)
                )
            })
            edge_src = t.edge_src
            edge_dst = t.edge_dst     
            random_indices = tf.random.shuffle(
                tf.range(tf.shape(edge_src)[0]))
            t_random = t.__class__(
                node_feature=t.node_feature,
                edge_src=tf.gather(edge_src, random_indices),
                edge_dst=tf.gather(edge_dst, random_indices),
                edge_feature=tf.gather(t.edge_feature, random_indices),
                edge_weight=tf.gather(t.edge_weight, random_indices),
            )
            t = t.update({
                'node_feature_residual': tf.random.uniform(t.node_feature.shape)})
            t = t.separate()

            residual = t.node_feature_residual

            t = t.remove(['node_feature_residual'])
            t = t.merge()

            test = tf.reduce_all(tf.equal(
                tf.math.round(t.propagate(
                    mode='mean', 
                    normalize=True, 
                    residual=residual, 
                    reduce='concat').node_feature, 3),
                tf.math.round(t_random.propagate(
                    mode='mean', 
                    normalize=True, 
                    residual=residual, 
                    reduce='concat').node_feature, 3)
            )).numpy()
            
            self.assertTrue(test)

    def test_ragged_normalize_propagate_nonragged_residual_reduce(self):
        for t in inputs:
            t = t.merge()
            t = t.update({
                'node_feature': tf.stack([
                    t.node_feature, 
                    t.node_feature + tf.random.uniform(t.node_feature.shape),
                    t.node_feature + tf.random.uniform(t.node_feature.shape),
                ], axis=1),
                'edge_weight': tf.random.uniform(
                    shape=(tf.shape(t.edge_src)[0], 3, 1)
                )
            })
            edge_src = t.edge_src
            edge_dst = t.edge_dst     
            random_indices = tf.random.shuffle(
                tf.range(tf.shape(edge_src)[0]))
            t_random = t.__class__(
                node_feature=t.node_feature,
                edge_src=tf.gather(edge_src, random_indices),
                edge_dst=tf.gather(edge_dst, random_indices),
                edge_feature=tf.gather(t.edge_feature, random_indices),
                edge_weight=tf.gather(t.edge_weight, random_indices),
            )

            residual = tf.random.uniform(t.node_feature.shape)
            t = t.separate()
            t_random = t_random.separate()
            
            test = tf.reduce_all(tf.equal(
                tf.math.round(t.propagate(
                    mode='mean', 
                    normalize=True, 
                    residual=residual, 
                    reduce='concat').merge().node_feature, 3),
                tf.math.round(t_random.propagate(
                    mode='mean', 
                    normalize=True, 
                    residual=residual, 
                    reduce='concat').merge().node_feature, 3)
            )).numpy()
            
            self.assertTrue(test)

    def test_indexing(self):
        inp = graph_tensor_1

        out = inp[:1]
        is_ragged = isinstance(out.node_feature, tf.RaggedTensor)
        is_ragged_spec = isinstance(out.spec.node_feature, tf.RaggedTensorSpec)
        self.assertTrue(is_ragged)
        self.assertTrue(is_ragged_spec)

        out = inp[0]
        is_not_ragged = not isinstance(out.node_feature, tf.RaggedTensor)
        is_not_ragged_spec = not isinstance(out.spec.node_feature, tf.RaggedTensorSpec)
        self.assertTrue(is_not_ragged)
        self.assertTrue(is_not_ragged_spec)

        out = inp[-1:]
        is_ragged = isinstance(out.node_feature, tf.RaggedTensor)
        is_ragged_spec = isinstance(out.spec.node_feature, tf.RaggedTensorSpec)
        self.assertTrue(is_ragged)
        self.assertTrue(is_ragged_spec)

        out = inp[1:3]
        test = tf.shape(out.node_feature) == tf.shape(graph_tensor_1_12.node_feature)
        self.assertTrue(test.numpy())

        inp_m = inp.merge()
        out_m = inp_m[1:3]
        test = tf.shape(out_m.node_feature) == tf.shape(graph_tensor_1_12.merge().node_feature)
        self.assertTrue(tf.reduce_all(test).numpy())

        test = tf.shape(out.node_feature) == tf.shape(out_m.separate().node_feature)
        self.assertTrue(test.numpy())
        test = tf.shape(out.merge().node_feature) == tf.shape(out_m.node_feature)
        self.assertTrue(tf.reduce_all(test).numpy())

        node_feature = inp['node_feature']
        test = tf.shape(node_feature) == tf.shape(inp.node_feature)
        self.assertTrue(test)

        inp = inp.update({'node_feature_2': inp.node_feature})
        node_feature_2 = inp['node_feature_2']
        test = tf.shape(node_feature_2) == tf.shape(inp.node_feature_2)
        self.assertTrue(test)

        out = inp[:100]
        out_m = inp_m[:100]
        test_1 = tf.shape(out.node_feature) == tf.shape(out_m.separate().node_feature)
        test_2 = tf.shape(out.merge().node_feature) == tf.shape(out_m.node_feature)
        self.assertTrue(test_1.numpy())
        self.assertTrue(tf.reduce_all(test_2).numpy())

        out = inp[100:]
        out_m = inp_m[100:]
        test_1 = tf.shape(out.node_feature) == tf.shape(out_m.separate().node_feature)
        test_2 = tf.shape(out.merge().node_feature) == tf.shape(out_m.node_feature)
        self.assertTrue(test_1.numpy())
        self.assertTrue(tf.reduce_all(test_2).numpy())

        try:
            node_feature_3 = inp[4]
            exception = None
        except Exception as e:
            exception = e
        self.assertIsInstance(exception, tf.errors.InvalidArgumentError)

        try:
            node_feature_3 = inp['node_feature_3']
            exception = None
        except Exception as e:
            exception = e
        self.assertIsInstance(exception, KeyError)
        
    def test_attributes(self):
        for t in inputs:
            t = t.update({'node_feature_2': t.node_feature})
            node_feature = t.node_feature
            node_feature_2 = t.node_feature_2
            test = tf.shape(node_feature) == tf.shape(node_feature_2)
            self.assertTrue(test.numpy())

            t = t.remove(['edge_feature'])
            edge_feature = t.edge_feature
            test = edge_feature is None 
            self.assertTrue(test)

            try:
                edge_feature_2 = t.edge_feature_2
                exception = None
            except Exception as e:
                exception = e
            self.assertIsInstance(exception, AttributeError)

    def test_spec(self):
        for t in inputs:
            t_spec = t.spec
            for s in t_spec._data_spec.values():
                test = isinstance(s, tf.RaggedTensorSpec)
                self.assertTrue(test)
                test = s.shape[0] is not None 
                self.assertTrue(test)
            
            test = t_spec.shape[0] is not None 
            self.assertTrue(test)

        for t in inputs:
            t = t.merge()
            t_spec = t.spec
            for s in t_spec._data_spec.values():
                test = isinstance(s, tf.TensorSpec)
                self.assertTrue(test)
                test = s.shape[0] is not None 
                self.assertTrue(test)
            
            test = t_spec.shape[0] is not None 
            self.assertTrue(test)

    def test_unspecific_spec(self):
        for t in inputs:
            t_spec = t.unspecific_spec
            for s in t_spec._data_spec.values():
                test = isinstance(s, tf.RaggedTensorSpec)
                self.assertTrue(test)
                test = s.shape[0] is None 
                self.assertTrue(test)
            
            test = t_spec.shape[0] is None 
            self.assertTrue(test)

        for t in inputs:
            t = t.merge()
            t_spec = t.unspecific_spec
            for s in t_spec._data_spec.values():
                test = isinstance(s, tf.TensorSpec)
                self.assertTrue(test)
                test = s.shape[0] is None 
                self.assertTrue(test)
            
            test = t_spec.shape[0] is None 
            self.assertTrue(test)

    def test_graph_tensor_spec(self):
        for tensor in inputs:
            try:
                tf.nest.assert_same_structure(tensor, tensor.spec)
                exception = None
            except Exception as e:
                exception = e 
            self.assertIsNone(exception)

            for t, s in zip(
                tensor._data.values(), 
                tensor.spec._data_spec.values()
            ):
                test1 = t.shape == s.shape
                test2 = t.dtype == s.dtype 
                test3 = t.ragged_rank == s.ragged_rank
                self.assertTrue(all([test1, test2, test3]))

            test1 = tensor.shape == tensor.spec.shape 
            test2 = tensor.dtype == tensor.spec.dtype 
            test3 = tensor.rank == tensor.spec.rank 
            self.assertTrue(all([test1, test2, test3]))

            tensor = tensor.merge()
            for t, s in zip(
                tensor._data.values(), 
                tensor.spec._data_spec.values()
            ):
                test1 = t.shape == s.shape
                test2 = t.dtype == s.dtype 
                self.assertTrue(all([test1, test2, test3]))

            test1 = tensor.shape == tensor.spec.shape 
            test2 = tensor.dtype == tensor.spec.dtype 
            test3 = tensor.rank == tensor.spec.rank 
            self.assertTrue(all([test1, test2, test3]))

    def test_dataset(self):
        y_batch = tf.constant([1., 2., 3., 4., 5., 6., 7.])
        x_batch = tf.concat(inputs, axis=0)
        ds = tf.data.Dataset.from_tensor_slices((x_batch, y_batch))
        ds = ds.batch(2).unbatch()

        for i, (x, y) in enumerate(ds.batch(1).map(lambda x, y: (x.merge(), y)).prefetch(-1)):
            test = tf.shape(x.node_feature) == tf.shape(x_batch[i].node_feature)
            self.assertTrue(tf.reduce_all(test).numpy())
        
        for x, y in ds.batch(2).take(1):
            test = tf.shape(x.node_feature) == tf.shape(x_batch[:2].node_feature)
            self.assertTrue(test.numpy())

    def test_graph_tensor_concat(self):
        out = tf.concat(inputs, axis=0)
        test = tf.shape(out.node_feature) == tf.shape(graph_tensor_123.node_feature)
        self.assertTrue(test.numpy())
        
        out = tf.concat([i.merge() for i in inputs], axis=0)
        test = tf.shape(out.node_feature) == tf.shape(graph_tensor_123.merge().node_feature)
        self.assertTrue(tf.reduce_all(test).numpy())

    def test_graph_tensor_stack(self):
        inp_flat = []
        for inp in inputs:
            for i in inp:
                inp_flat.append(i)
        out = tf.stack(inp_flat, axis=0)
        test = tf.shape(out.node_feature) == tf.shape(graph_tensor_123.node_feature)
        self.assertTrue(test)
        
    def test_graph_tensor_boolean_mask(self):

        gt = graph_tensor_1[2]
        node_feature_expected = tf.constant([
            [1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.],
            [0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 1.],
            [0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.]
        ], dtype=tf.float32)
        edge_feature_expected = tf.zeros((0, 5), dtype=tf.float32)
        edge_src_expected = tf.zeros((0,), dtype=tf.int32)
        edge_dst_expected = tf.zeros((0,), dtype=tf.int32)
        mask = tf.constant([
            True, True, False, True, False, False, False, True])
        gt_masked = tf.boolean_mask(gt, mask, axis='node')
        test1 = tf.reduce_all(gt_masked.node_feature == node_feature_expected).numpy()
        test2 = tf.reduce_all(gt_masked.edge_feature == edge_feature_expected).numpy()
        test3 = tf.reduce_all(gt_masked.edge_src == edge_src_expected).numpy()
        test4 = tf.reduce_all(gt_masked.edge_dst == edge_dst_expected).numpy()
        self.assertTrue(all([test1, test2, test3, test4]))

        gt = graph_tensor_1[2]
        node_feature_expected = tf.constant([
            [0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 1.],
            [0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        ], dtype=tf.float32)
        edge_feature_expected = tf.constant([
            [0., 0., 1., 0., 0.],
            [0., 0., 1., 0., 0.],
            [1., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0.],
        ],  dtype=tf.float32)
        edge_src_expected = tf.constant(
            [0, 1, 1, 1, 2, 3],  dtype=tf.int32)
        edge_dst_expected = tf.constant(
            [1, 0, 2, 3, 1, 1],  dtype=tf.int32)
        mask = tf.constant([
            False, True, True, True, False, False, False, True])
        gt_masked = tf.boolean_mask(gt, mask, axis='node')
        test1 = tf.reduce_all(gt_masked.node_feature == node_feature_expected).numpy()
        test2 = tf.reduce_all(gt_masked.edge_feature == edge_feature_expected).numpy()
        test3 = tf.reduce_all(gt_masked.edge_src == edge_src_expected).numpy()
        test4 = tf.reduce_all(gt_masked.edge_dst == edge_dst_expected).numpy()
        self.assertTrue(all([test1, test2, test3, test4]))

        gt = graph_tensor_5.merge()
        node_feature_expected = tf.constant(
            [[0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
             [0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
             [1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.],
             [0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 1.],
             [0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
             [0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
             [0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
             [0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
             [0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.]], dtype=tf.float32)
        edge_feature_expected = tf.constant(
            [[0., 0., 1., 0., 0.],
             [0., 0., 1., 0., 0.],
             [0., 0., 1., 0., 0.],
             [0., 0., 1., 0., 0.],
             [1., 0., 0., 0., 0.],
             [1., 0., 0., 0., 0.],
             [1., 0., 0., 0., 0.],
             [1., 0., 0., 0., 0.]], dtype=tf.float32)
        edge_src_expected = tf.constant([
             0, 1, 3, 4, 4, 4, 5, 7], dtype=tf.int32)
        edge_dst_expected = tf.constant([
             1, 0, 4, 3, 5, 7, 4, 4], dtype=tf.int32)
        mask = tf.constant([
            True, True, False, True, True, True, True, False, True, False, True, True])
        gt_masked = tf.boolean_mask(gt, mask, axis='node')
        test1 = tf.reduce_all(gt_masked.node_feature == node_feature_expected).numpy()
        test2 = tf.reduce_all(gt_masked.edge_feature == edge_feature_expected).numpy()
        test3 = tf.reduce_all(gt_masked.edge_src == edge_src_expected).numpy()
        test4 = tf.reduce_all(gt_masked.edge_dst == edge_dst_expected).numpy()
        self.assertTrue(all([test1, test2, test3, test4]))

        gt = graph_tensor_5.merge()
        node_feature_expected = tf.constant(
            [[0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
             [0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
             [0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
             [1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.],
             [0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 1.],
             [0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
             [0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
             [0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
             [0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
             [0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
             [0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
             [0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.]], dtype=tf.float32)
        edge_feature_expected = tf.constant(
            [[0., 0., 1., 0., 0.],
             [0., 0., 1., 0., 0.],
             [0., 0., 1., 0., 0.],
             [0., 0., 1., 0., 0.],
             [1., 0., 0., 0., 0.],
             [1., 0., 0., 0., 0.],
             [1., 0., 0., 0., 0.],
             [1., 0., 0., 0., 0.],
             [1., 0., 0., 0., 0.],
             [1., 0., 0., 0., 0.],
             [1., 0., 0., 0., 0.]], dtype=tf.float32)
        edge_src_expected = tf.constant([
            1, 1, 4, 5,  5, 6, 7, 8,  9, 10, 10], dtype=tf.int32)
        edge_dst_expected = tf.constant([
            0, 2, 5, 4, 10, 7, 6, 7, 10,  5,  9], dtype=tf.int32)
        mask = tf.constant([
            False, True, True, False, 
            True, True, False, True, False, True, True, False, True, False, False, True, True, True])
        gt_masked = tf.boolean_mask(gt, mask, axis='edge')
        test1 = tf.reduce_all(gt_masked.node_feature == node_feature_expected).numpy()
        test2 = tf.reduce_all(gt_masked.edge_feature == edge_feature_expected).numpy()
        test3 = tf.reduce_all(gt_masked.edge_src == edge_src_expected).numpy()
        test4 = tf.reduce_all(gt_masked.edge_dst == edge_dst_expected).numpy()
        self.assertTrue(all([test1, test2, test3, test4]))

    def test_graph_tensor_shape(self):
        for t in inputs:
            # tf shape of graph tensor in ragged state correspond to 
            # tf shape of graph tensor in non-ragged state. This is possibly
            # a temporary fix to allow keras.Model.predict to concatenate
            # ragged graph tensors.
            test = tf.shape(t) == tf.shape(t.merge().node_feature)
            self.assertTrue(tf.reduce_all(test).numpy())

        for t in inputs:
            t = t.merge()
            test = tf.shape(t) == tf.shape(t.node_feature)
            self.assertTrue(tf.reduce_all(test).numpy())
        
    def test_graph_tensor_gather(self):
        for t in inputs:
            t = t.merge()
            t_res = tf.matmul(t, tf.transpose(t.node_feature))
            res = tf.matmul(t.node_feature, tf.transpose(t.node_feature))
            test = t_res == res
            self.assertTrue(tf.reduce_all(test).numpy())

    def test_graph_tensor_matmul(self):
        for t in inputs:
            t = t.merge()
            t_res = tf.matmul(t, tf.transpose(t.node_feature))
            res = tf.matmul(t.node_feature, tf.transpose(t.node_feature))
            test = t_res == res
            self.assertTrue(tf.reduce_all(test).numpy())

    def test_graph_tensor_add(self):
        for t in inputs:
            t_add = tf.math.add(t, 10.)
            test = t_add.node_feature == t.node_feature + 10.
            self.assertTrue(tf.reduce_all(test).numpy())

        for t in inputs:
            t = t.merge()
            t_add = tf.math.add(t, 10.)
            test = t_add.node_feature == t.node_feature + 10.
            self.assertTrue(tf.reduce_all(test).numpy())

    def test_graph_tensor_add_sub_abs(self):

        for t in inputs:
            t = t.merge()
            t = t.update({'node_feature': tf.zeros_like(t.node_feature)})
            t = t.separate()
            t_sub = tf.math.subtract(t, -10.)
            t_add = tf.math.add(t, 10.)
            t_abs = tf.math.abs(t_sub)
            test = t_abs.node_feature == t_add.node_feature
            self.assertTrue(tf.reduce_all(test).numpy())

        for t in inputs:
            t = t.merge()
            t = t.update({'node_feature': tf.zeros_like(t.node_feature)})
            t_sub = tf.math.subtract(t, -10.)
            t_add = tf.math.add(t, 10.)
            t_abs = tf.math.abs(t_sub)
            test = t_abs.node_feature == t_add.node_feature
            self.assertTrue(tf.reduce_all(test).numpy())


if __name__ == "__main__":
    unittest.main()