import tensorflow as tf
import numpy as np
from dataclasses import dataclass 

from molgraph import GraphTensor
from molgraph.chemistry import MolecularGraphEncoder
from molgraph.applications.proteomics.peptide import Peptide
from molgraph.applications.proteomics.definitions import _num_residue_types
from molgraph.applications.proteomics.definitions import _residue_node_mask
from molgraph.applications.proteomics.definitions import _residue_indicator


@dataclass
class PeptideGraphEncoder(MolecularGraphEncoder):

    super_nodes: bool = True

    def call(self, sequence: str, index_dtype: str = 'int32') -> GraphTensor:
        peptide = Peptide(sequence)
        x = super().call(peptide.smiles)
        residue_sizes = peptide.residue_sizes

        if not self.super_nodes:
            subgraph_indicator = np.repeat(
                np.arange(len(residue_sizes)), residue_sizes)
            return x.update({_residue_indicator: subgraph_indicator})
        
        num_nodes = x.node_feature.shape[0]
        num_super_nodes = len(peptide)
        num_super_edges = sum(residue_sizes)
        if self.self_loops:
            num_super_edges += len(peptide) 
        
        data = {
            _residue_node_mask: np.concatenate([[0] * num_nodes, [1] * num_super_nodes])
        }
        
        residue_index = np.arange(num_nodes, num_nodes + num_super_nodes)
        super_target_index = np.repeat(residue_index, residue_sizes)
        super_source_index = np.arange(num_nodes)

        data['node_feature'] = np.pad(x.node_feature, [(0, num_super_nodes), (0, _num_residue_types)])
        data['node_feature'][-num_super_nodes:, -_num_residue_types:] = np.eye(_num_residue_types)[peptide.residue_indices]
        data['edge_src'] = np.concatenate([x.edge_src, super_source_index]).astype(index_dtype)
        data['edge_dst'] = np.concatenate([x.edge_dst, super_target_index]).astype(index_dtype)
        if self.self_loops:
            data['edge_src'] = np.concatenate([data['edge_src'], residue_index])
            data['edge_dst'] = np.concatenate([data['edge_dst'], residue_index])
            data['edge_feature'] = np.pad(x.edge_feature, [(0, num_super_edges), (0, 2)])
            data['edge_feature'][-num_super_edges:, -2:] = np.eye(2)[np.concatenate([[0] * sum(residue_sizes), [1] * num_super_nodes])]
        else:
            data['edge_feature'] = np.pad(x.edge_feature, [(0, num_super_edges), (0, 1)])
            data['edge_feature'][-num_super_edges:, -1:] = 1.0
        return GraphTensor(**data)
    
    def __call__(self, inputs, **kwargs):
        tensor = super().__call__(inputs, **kwargs)
        if self.super_nodes:
            return tensor 
        # Increment residue indicator as the graph encode many peptides:
        residue_sizes = tf.math.segment_max(
            getattr(tensor, _residue_indicator), tensor.graph_indicator) + 1
        incr = tf.repeat(
            tf.concat([[0], tf.cumsum(residue_sizes)[:-1]], axis=0), tensor.sizes)
        return tensor.update({
            _residue_indicator: getattr(tensor, _residue_indicator) + incr})


@dataclass
class _BondlessPeptideGraphEncoder(MolecularGraphEncoder):

    """Temporary and for experimental purposes only"""

    def call(self, sequence: str, index_dtype: str = 'int32') -> GraphTensor:
        
        peptide = Peptide(sequence)
        
        x = super().call(peptide.smiles)
        
        residue_sizes = peptide.residue_sizes
        num_nodes = x.node_feature.shape[0]
        num_super_nodes = len(peptide)
        
        data = {
            _residue_node_mask: np.concatenate([[0] * num_nodes, [1] * num_super_nodes])
        }

        residue_index = np.arange(num_nodes, num_nodes + num_super_nodes)
        super_target_index = np.repeat(residue_index, residue_sizes)
        super_source_index = np.arange(num_nodes)

        data['node_feature'] = np.pad(x.node_feature, [(0, num_super_nodes), (0, _num_residue_types)])
        data['node_feature'][-num_super_nodes:, -_num_residue_types:] = np.eye(_num_residue_types)[peptide.residue_indices]
        data['edge_src'] = np.concatenate([super_source_index, residue_index]).astype(index_dtype)
        data['edge_dst'] = np.concatenate([super_target_index, residue_index]).astype(index_dtype)
        data['edge_feature'] = np.eye(2)[np.concatenate([[0] * sum(residue_sizes), [1] * num_super_nodes])].astype(np.float32)

        return GraphTensor(**data)




