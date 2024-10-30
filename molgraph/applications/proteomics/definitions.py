import json 
from pathlib import Path 


data_dir = Path(__file__).parent / 'data'

with open(data_dir / '_aa_index.json', 'r') as json_file:
    _residue_index: dict = json.load(json_file)

with open(data_dir / '_aa_smiles.json', 'r') as json_file:
    _residue_smiles: dict = json.load(json_file)

_num_residue_types = len(_residue_index)
_residue_node_mask = 'node_super_mask'          # when using super nodes
_residue_indicator = 'node_subgraph_indicator'  # when not using super nodes

