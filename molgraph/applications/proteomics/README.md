# Proteomics

> Ongoing project, subject to changes in the future.

## Brief example

```python
from molgraph import chemistry
from molgraph.applications import proteomics

atom_encoder = chemistry.Featurizer([
    chemistry.features.Symbol({'C', 'N', 'O', 'S', 'P'}, oov_size=1)
])
bond_encoder = chemistry.Featurizer([
    chemistry.features.BondType({'AROMATIC', 'TRIPLE', 'DOUBLE', 'SINGLE'}, oov_size=1),
])
encoder = proteomics.PeptideGraphEncoder(atom_encoder, bond_encoder, super_nodes=True) 

# Example input:
peptide_graph = encoder(['CYIQNCPLG'])
# Create model (keras.Sequential instance)
model = proteomics.PeptideModel(config=None, spec=peptide_graph.spec)
# Predict input
prediction = model(peptide_graph)

# generate more inputs ...
# train model on inputs ...

saliency = proteomics.PeptideSaliency(model)

# Will include saliency values of super nodes if `super_nodes=True`
saliency_values = saliency(peptide_graph.separate())
```

Add your own residue SMILES (within a session):

```python
from molgraph import chemistry
from molgraph.applications import proteomics

# Note: Arginine exists by default, and is only added below for 
#       illustration. Inputted Arginine SMILES below is not acceptable 
#       as it cannot be concatenated with other SMILES. Specifying 
#       canonicalize=True, should make the Arginine SMILES concatenateable.
proteomics.Peptide.register_residue_smiles(
    {
        "N[Deamidated]": "N[C@@H](CC(O)=O)C(=O)O",
        "Q[Deamidated]": "N[C@@H](CCC(=O)O)C(=O)O",
        "R": "C(C[C@@H](C(=O)O)N)CN=C(N)N" 
    }, 
    canonicalize=True
)

encoder = proteomics.PeptideGraphEncoder(
    chemistry.Featurizer([
        chemistry.features.Symbol({'C', 'N', 'O', 'S', 'P'}, oov_size=1)
    ]),
    chemistry.Featurizer([
        chemistry.features.BondType({'AROMATIC', 'DOUBLE', 'SINGLE'}, oov_size=1),
    ]),
    super_nodes=True
) 

graph_tensor = encoder('AAAN[Deamidated]Q[Deamidated]GGG')
```
