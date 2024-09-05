import json 
from pathlib import Path 


data_dir = Path(__file__).parent / 'data'

with open(data_dir / '_aa_index.json', 'r') as json_file:
    _residue_index = json.load(json_file)

with open(data_dir / '_aa_smiles.json', 'r') as json_file:
    _residue_smiles = json.load(json_file)

_num_residue_types = len(_residue_index)
_residue_node_indicator = 'node_super_indicator'


