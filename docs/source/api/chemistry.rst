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

Featurizer
==================
.. autoclass:: molgraph.chemistry.AtomicFeaturizer()
  :members:  __call__,

Tokenizer
==================
.. autoclass:: molgraph.chemistry.AtomicTokenizer()
  :members: __call__,

Features
==================

.. autoclass:: molgraph.chemistry.AtomicFeature()
  :members: __call__

.. autoclass:: molgraph.chemistry.features.Symbol(AtomicFeature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.Hybridization(AtomicFeature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.CIPCode(AtomicFeature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.ChiralCenter(AtomicFeature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.FormalCharge(AtomicFeature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.TotalNumHs(AtomicFeature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.TotalValence(AtomicFeature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.NumRadicalElectrons(AtomicFeature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.Degree(AtomicFeature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.Aromatic(AtomicFeature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.Hetero(AtomicFeature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.HydrogenDonor(AtomicFeature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.HydrogenAcceptor(AtomicFeature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.RingSize(AtomicFeature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.Ring(AtomicFeature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.CrippenLogPContribution(AtomicFeature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.CrippenMolarRefractivityContribution(AtomicFeature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.TPSAContribution(AtomicFeature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.LabuteASAContribution(AtomicFeature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.GasteigerCharge(AtomicFeature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.BondType(AtomicFeature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.Conjugated(AtomicFeature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.Rotatable(AtomicFeature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.Stereo(AtomicFeature)
  :members: __call__


**********************
Chemistry ops
**********************
.. automodule:: molgraph.chemistry.ops
  :members: 
  :member-order: bysource


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
