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
.. autoclass:: molgraph.chemistry.Featurizer()
  :members:  __call__,

Tokenizer
==================
.. autoclass:: molgraph.chemistry.Tokenizer()
  :members: __call__,

Features
==================

.. autoclass:: molgraph.chemistry.Feature()
  :members: __call__

.. autoclass:: molgraph.chemistry.features.Symbol(Feature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.Hybridization(Feature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.CIPCode(Feature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.ChiralCenter(Feature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.FormalCharge(Feature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.TotalNumHs(Feature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.TotalValence(Feature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.NumRadicalElectrons(Feature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.Degree(Feature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.Aromatic(Feature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.Hetero(Feature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.HydrogenDonor(Feature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.HydrogenAcceptor(Feature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.RingSize(Feature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.Ring(Feature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.CrippenLogPContribution(Feature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.CrippenMolarRefractivityContribution(Feature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.TPSAContribution(Feature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.LabuteASAContribution(Feature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.GasteigerCharge(Feature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.BondType(Feature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.Conjugated(Feature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.Rotatable(Feature)
  :members: __call__

.. autoclass:: molgraph.chemistry.features.Stereo(Feature)
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
