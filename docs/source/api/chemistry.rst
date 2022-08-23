####################
Chemistry
####################

********************
Molecular
********************

MolecularGraphEncoder
===========================
.. autoclass:: molgraph.chemistry.MolecularGraphEncoder(molgraph.chemistry.BaseMolecularGraphEncoder)
  :special-members: __call__

MolecularGraphEncoder3D
===========================
.. autoclass:: molgraph.chemistry.MolecularGraphEncoder3D(molgraph.chemistry.BaseMolecularGraphEncoder)
  :special-members: __call__

ConformerGenerator
===========================
.. autoclass:: molgraph.chemistry.ConformerGenerator()
  :members: available_embedding_methods, available_force_field_methods, __call__

********************
Atomic
********************

Featurizers
==================
.. autoclass:: molgraph.chemistry.AtomFeaturizer(molgraph.chemistry.AtomicFeaturizer)
  :members:  __call__, encode_atoms,

.. autoclass:: molgraph.chemistry.BondFeaturizer(molgraph.chemistry.AtomicFeaturizer)
  :members: __call__, encode_bonds,

Tokenizers
==================
.. autoclass:: molgraph.chemistry.AtomTokenizer(molgraph.chemistry.AtomicTokenizer)
  :members: __call__, encode_atoms,

.. autoclass:: molgraph.chemistry.BondTokenizer(molgraph.chemistry.AtomicTokenizer)
  :members: __call__, encode_bonds,

Features
==================

.. automodule:: molgraph.chemistry.atomic.features
  :members:
  :member-order: bysource
  :special-members: __call__

**********************
TF records
**********************

.. automodule:: molgraph.chemistry.benchmark.tf_records
  :members: write, load,
  :member-order: bysource

**********************
Datasets
**********************
.. autoclass:: molgraph.chemistry.benchmark.datasets.DatasetFactory
  :members: get, get_config, registered_datasets,
  :member-order: bysource
