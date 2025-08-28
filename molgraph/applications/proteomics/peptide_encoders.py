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
    bidirectional_super_edges: bool = True 

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
        if self.bidirectional_super_edges:
            num_super_edges += sum(residue_sizes)
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
        data['edge_src'] = np.concatenate([x.edge_src, super_source_index])
        data['edge_dst'] = np.concatenate([x.edge_dst, super_target_index])

        if self.bidirectional_super_edges:
            data['edge_src'] = np.concatenate([data['edge_src'], super_target_index])
            data['edge_dst'] = np.concatenate([data['edge_dst'], super_source_index])

        edge_feature_pad = 1 + int(self.bidirectional_super_edges) + int(self.self_loops)

        data['edge_feature'] = np.pad(x.edge_feature, [(0, num_super_edges), (0, edge_feature_pad)])

        pad_index = 0
        pad_indices = np.array([pad_index] * num_nodes)
        if self.bidirectional_super_edges:
            pad_index += 1
            pad_indices = np.concatenate([pad_indices, [pad_index] * num_nodes])
        if self.self_loops:
            pad_index += 1
            pad_indices = np.concatenate([pad_indices, [pad_index] * num_super_nodes])
            data['edge_src'] = np.concatenate([data['edge_src'], residue_index])
            data['edge_dst'] = np.concatenate([data['edge_dst'], residue_index])

        data['edge_feature'][-num_super_edges:, -edge_feature_pad:] = np.eye(edge_feature_pad)[pad_indices]
        data['edge_src'] = data['edge_src'].astype(index_dtype)
        data['edge_dst'] = data['edge_dst'].astype(index_dtype)
        return GraphTensor(**data)
    




