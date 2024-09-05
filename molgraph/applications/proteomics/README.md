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
encoder = proteomics.PeptideGraphEncoder(atom_encoder, bond_encoder) 

# Example input:
peptide_graph = encoder(['CYIQNCPLG'])
# Create model (keras.Sequential instance)
model = proteomics.PepGNN(config=None, spec=peptide_graph.spec)
# Predict input
prediction = model(peptide_graph)

# generate more inputs ...
# train model on inputs ...

saliency = proteomics.PeptideSaliency(model)

saliency_values = saliency(peptide_graph)
```