import molgraph
from molgraph.chemistry import MolecularGraphEncoder
from molgraph.chemistry import Featurizer
from molgraph.chemistry import features

# Define atomic encoders
atom_encoder = Featurizer([
    features.Symbol({'C', 'N', 'O'}, oov_size=1),
    features.Hybridization({'SP', 'SP2', 'SP3'}, oov_size=1),
    features.HydrogenDonor(),
    features.HydrogenAcceptor(),
    features.Hetero()
])
bond_encoder = Featurizer([
    features.BondType({'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'}),
    features.Rotatable()
])
# Define molecular graph encoder
encoder = MolecularGraphEncoder(atom_encoder, bond_encoder)

graph_tensor_1 = encoder([
    'OCC1OC(C(C1O)O)n1cnc2c1ncnc2N', 'C(C(=O)O)N', '[Na+].[O-]c1ccccc1']).separate()
graph_tensor_2 = encoder([
    'C']).separate()
graph_tensor_3 = encoder([
    'CC', 'C', 'CC']).separate()
graph_tensor_4 = encoder([
    'OCC1OC(C(C1O)O)n1cnc2c1ncnc2N', 'C(C(=O)O)N', '[Na+].[O-]c1ccccc1'],
    index_dtype='int64').separate()
graph_tensor_5 = encoder([
    'CCC', '[Na+].[O-]c1ccccc1', 'C']).separate()

graph_tensor_1_12 = encoder([
    'C(C(=O)O)N', '[Na+].[O-]c1ccccc1']).separate()

graph_tensor_123 = encoder([
    'OCC1OC(C(C1O)O)n1cnc2c1ncnc2N', 
    'C(C(=O)O)N', 
    '[Na+].[O-]c1ccccc1',
    'C',
    'CC', 
    'C', 
    'CC'
]).separate()