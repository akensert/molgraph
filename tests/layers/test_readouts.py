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


if __name__ == "__main__":
    unittest.main()
