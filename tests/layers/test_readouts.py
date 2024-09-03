from tests.layers._common import inputs

import unittest

import tensorflow as tf

from molgraph.layers import readout, attentional, message_passing


class TestReadout(unittest.TestCase):

    def test_segment_pool(self):
        for x in inputs:
            y = readout.segment_pool.SegmentPoolingReadout()(x)
            expected_shape = x.shape[:1].concatenate(x.node_feature.shape[-1:])
            test = y.shape == expected_shape
            self.assertTrue(test)

    def test_set_gather(self):
        for x in inputs:
            y = readout.set_gather.SetGatherReadout()(x)
            expected_shape = x.shape[:1].concatenate(x.node_feature.shape[-1] * 2)
            test = y.shape == expected_shape
            self.assertTrue(test)

    def test_transformer_encoder(self):
        for x in inputs:
            y = readout.transformer_encoder.TransformerEncoderReadout(128)(x)
            expected_shape = x.shape[:1].concatenate(x.node_feature.shape[-1:])
            test = y.shape == expected_shape
            self.assertTrue(test)

    def test_attentive_fp_readout(self):
        for x in inputs:
            x = attentional.gat_conv.GATConv(32)(x)
            y = readout.attentive_fp_readout.AttentiveFPReadout()(x)
            expected_shape = x.shape[:1].concatenate(x.node_feature.shape[-1:])
            test = y.shape == expected_shape
            self.assertTrue(test)

    def test_node_readout(self):
        for i, x in enumerate(inputs):
            if i == 4 or i == 5:
                continue
            x = message_passing.edge_conv.EdgeConv(32)(x)
            y = readout.node_readout.NodeReadout()(x)
            if y.is_ragged():
                y = y.merge()
                x = x.merge()
            expected_shape = x.node_feature.shape[:1].concatenate(x.edge_state.shape[-1:])
            test = y.node_feature.shape == expected_shape
            self.assertTrue(test)

    def test_super_node_readout(self):
        x = inputs[1]

        i = tf.random.uniform(x.node_feature.shape[:1], 0, 2, tf.dtypes.int32)
        x1 = x.update({'node_super_indicator': i})
        x2 = x1.separate()

        y1 = readout.super_node_readout.SuperNodeReadout('node_super_indicator')(x1)
        y2 = readout.super_node_readout.SuperNodeReadout('node_super_indicator')(x2)

        n_super = int(tf.math.reduce_max(tf.reduce_sum(x2.node_super_indicator, axis=1)))

        test1 = y1.shape == (2, n_super, 12)
        test2 = y2.shape == (2, n_super, 12)

        self.assertTrue(test1)
        self.assertTrue(test2)



if __name__ == "__main__":
    unittest.main()
