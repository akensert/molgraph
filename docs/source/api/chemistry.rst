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

Atom features
------------------
.. autoclass:: molgraph.chemistry.features.Symbol(molgraph.chemistry.features.AtomicFeature)
  :special-members: __init__
  :undoc-members: __init__

.. autoclass:: molgraph.chemistry.features.Hybridization(molgraph.chemistry.features.AtomicFeature)
  :special-members: __init__
  :undoc-members: __init__

.. autoclass:: molgraph.chemistry.features.CIPCode(molgraph.chemistry.features.AtomicFeature)
  :special-members: __init__
  :undoc-members: __init__

.. autoclass:: molgraph.chemistry.features.ChiralCenter(molgraph.chemistry.features.AtomicFeature)
  :special-members: __init__
  :undoc-members: __init__

.. autoclass:: molgraph.chemistry.features.FormalCharge(molgraph.chemistry.features.AtomicFeature)
  :special-members: __init__
  :undoc-members: __init__

.. autoclass:: molgraph.chemistry.features.TotalNumHs(molgraph.chemistry.features.AtomicFeature)
  :special-members: __init__
  :undoc-members: __init__

.. autoclass:: molgraph.chemistry.features.TotalValence(molgraph.chemistry.features.AtomicFeature)
  :special-members: __init__
  :undoc-members: __init__

.. autoclass:: molgraph.chemistry.features.NumRadicalElectrons(molgraph.chemistry.features.AtomicFeature)
  :special-members: __init__
  :undoc-members: __init__

.. autoclass:: molgraph.chemistry.features.Degree(molgraph.chemistry.features.AtomicFeature)
  :special-members: __init__
  :undoc-members: __init__

.. autoclass:: molgraph.chemistry.features.Aromatic(molgraph.chemistry.features.AtomicFeature)
  :special-members: __init__
  :undoc-members: __init__

.. autoclass:: molgraph.chemistry.features.Hetero(molgraph.chemistry.features.AtomicFeature)
  :special-members: __init__
  :undoc-members: __init__

.. autoclass:: molgraph.chemistry.features.HydrogenDonor(molgraph.chemistry.features.AtomicFeature)
  :special-members: __init__
  :undoc-members: __init__

.. autoclass:: molgraph.chemistry.features.HydrogenAcceptor(molgraph.chemistry.features.AtomicFeature)
  :special-members: __init__
  :undoc-members: __init__

.. autoclass:: molgraph.chemistry.features.RingSize(molgraph.chemistry.features.AtomicFeature)
  :special-members: __init__
  :undoc-members: __init__

.. autoclass:: molgraph.chemistry.features.Ring(molgraph.chemistry.features.AtomicFeature)
  :special-members: __init__
  :undoc-members: __init__

.. autoclass:: molgraph.chemistry.features.CrippenLogPContribution(molgraph.chemistry.features.AtomicFeature)
  :special-members: __init__
  :undoc-members: __init__

.. autoclass:: molgraph.chemistry.features.CrippenMolarRefractivityContribution(molgraph.chemistry.features.AtomicFeature)
  :special-members: __init__
  :undoc-members: __init__

.. autoclass:: molgraph.chemistry.features.TPSAContribution(molgraph.chemistry.features.AtomicFeature)
  :special-members: __init__
  :undoc-members: __init__

.. autoclass:: molgraph.chemistry.features.LabuteASAContribution(molgraph.chemistry.features.AtomicFeature)
  :special-members: __init__
  :undoc-members: __init__

.. autoclass:: molgraph.chemistry.features.GasteigerCharge(molgraph.chemistry.features.AtomicFeature)
  :special-members: __init__
  :undoc-members: __init__


Bond features
------------------
.. autoclass:: molgraph.chemistry.features.BondType(molgraph.chemistry.features.AtomicFeature)
  :special-members: __init__
  :undoc-members: __init__

.. autoclass:: molgraph.chemistry.features.Conjugated(molgraph.chemistry.features.AtomicFeature)
  :special-members: __init__
  :undoc-members: __init__

.. autoclass:: molgraph.chemistry.features.Rotatable(molgraph.chemistry.features.AtomicFeature)
  :special-members: __init__
  :undoc-members: __init__

.. autoclass:: molgraph.chemistry.features.Stereo(molgraph.chemistry.features.AtomicFeature)
  :special-members: __init__
  :undoc-members: __init__
