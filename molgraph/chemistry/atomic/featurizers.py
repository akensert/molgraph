import numpy as np
from dataclasses import dataclass
from dataclasses import field
from typing import Union
from typing import List
from typing import Any
from typing import Dict
from rdkit import Chem
from abc import abstractmethod
from abc import ABC

from molgraph.chemistry.atomic.features import AtomicFeature
from molgraph.chemistry.atomic.encoders import AtomicEncoder
from molgraph.chemistry.atomic.encoders import FloatEncoder
from molgraph.chemistry.atomic.encoders import OrdinalEncoder
from molgraph.chemistry.atomic.encoders import NominalEncoder



@dataclass
class AtomicFeaturizer:

    features: List[AtomicFeature]
    dtype: Union[str, np.dtype] = 'float32'
    ndim: int = field(init=False)

    def __post_init__(self) -> None:
        self.features = [_encoder_from_feature(f) for f in self.features]
        self.ndim = _get_atom_or_bond_dim(self)

    def __call__(self, inputs: Union[Chem.Atom, Chem.Bond]) -> np.ndarray:
        '''Featurizes a single rdkit Atom or Bond.

        Args:
            inputs (rdkit.Chem.Atom, rdkit.Chem.Bond):
                A single RDKit atom or bond.

        Returns:
            np.ndarray: A numberical encoding of an atom or a bond.
        '''
        return np.concatenate([feature(inputs) for feature in self.features])


class AtomFeaturizer(AtomicFeaturizer):
    '''Atom featurizer.

    Args:
        features (list[AtomicFeature]):
            List of atom features.
        dtype (str, np.dtype):
            The output dtype.

    **Example:**

    >>> atom_featurizer = molgraph.chemistry.AtomFeaturizer([
    ...     molgraph.chemistry.features.Symbol(
    ...         allowable_set={'C', 'N'},           # will get sorted internally
    ...         ordinal=False,
    ...         oov_size=1                          # OOVs are preprended
    ...     ),
    ...     molgraph.chemistry.features.Hybridization(
    ...         allowable_set={'SP', 'SP2', 'SP3'},
    ...         ordinal=False,
    ...         oov_size=1
    ...     )
    ... ])
    >>> # Obtain an Atom
    >>> rdkit_mol = rdkit.Chem.MolFromSmiles('CC')
    >>> rdkit_atom = rdkit_mol.GetAtomWithIdx(0)
    >>> # Encode Atom as a numerical vector
    >>> atom_featurizer(rdkit_atom)
    array([0., 1., 0., 0., 0., 0., 1.], dtype=float32)
    '''

    def encode_atoms(self, inputs: List[Chem.Atom]) -> np.ndarray:
        '''Featurizes a list of RDKit atoms (rdkit.Chem.Atom).

        Args:
            inputs (list[rdkit.Chem.Atom]):
                List of RDKit atoms.

        Returns:
            np.ndarray: Numerical encodings of multiple atoms.
        '''
        atom_features = []
        for atom in inputs:
            _check_atom(atom)
            atom_features.append(self(atom))
        return np.asarray(atom_features, dtype=self.dtype)


class BondFeaturizer(AtomicFeaturizer):
    '''Bond featurizer.

    Args:
        features (list[AtomicFeature]):
            List of bond features.
        dtype (str, np.dtype):
            The output dtype.

    **Example:**

    >>> bond_featurizer = molgraph.chemistry.BondFeaturizer([
    ...     molgraph.chemistry.features.BondType(
    ...         allowable_set={'SINGLE', 'DOUBLE'}, # will get sorted internally
    ...         ordinal=False,
    ...         oov_size=1                          # OOVs are prepended
    ...     ),
    ... ])
    >>> # Obtain a Bond
    >>> rdkit_mol = rdkit.Chem.MolFromSmiles('CC')
    >>> rdkit_bond = rdkit_mol.GetBondWithIdx(0)
    >>> # Encode Bond as a numerical vector
    >>> bond_featurizer(rdkit_bond)
    array([0., 0., 1.], dtype=float32)
    '''

    def encode_bonds(
        self,
        inputs: List[Chem.Bond],
        self_loops: bool = False,
    ) -> np.ndarray:
        '''Featurizes a list of RDKit bonds (rdkit.Chem.Bond).

        Args:
            inputs (list[rdkit.Chem.Bond]):
                List of RDKit bonds.

        Returns:
            np.ndarray: Numerical encodings of multiple bonds.
        '''

        if self_loops:
            self.ndim += 1

        if not len(inputs):
            return np.zeros([0, self.ndim]).astype(self.dtype)

        bond_features = []
        for bond in inputs:
            _check_bond(bond)
            if bond is None:
                # `None` indicates that bond is a self loop
                encoding = np.zeros(self.ndim, dtype=self.dtype)
                encoding[-1] = 1
            else:
                encoding = self(bond)
                if len(encoding) < self.ndim:
                    # only necessary if `self_loops` is set to True
                    encoding = np.concatenate([encoding, [0]])

            bond_features.append(encoding)

        return np.asarray(bond_features, dtype=self.dtype)


def _encoder_from_feature(inputs: AtomicFeature) -> AtomicEncoder:
    if not hasattr(inputs, 'allowable_set'):
        return FloatEncoder(inputs)
    elif getattr(inputs, 'ordinal', False):
        return OrdinalEncoder(inputs)
    else:
        return NominalEncoder(inputs)

def _get_atom_dim(atom_encoder: AtomFeaturizer) -> Union[int, None]:
    if atom_encoder is not None:
        dummy_mol = Chem.MolFromSmiles('CC')
        return len(atom_encoder(dummy_mol.GetAtoms()[0]))
    return None

def _get_bond_dim(bond_encoder: BondFeaturizer) -> Union[int, None]:
    if bond_encoder is not None:
        dummy_mol = Chem.MolFromSmiles('CC')
        return len(bond_encoder(dummy_mol.GetBonds()[0]))
    return None

def _get_atom_or_bond_dim(atomic_encoder: Union[AtomFeaturizer, BondFeaturizer]):
    if isinstance(atomic_encoder, AtomFeaturizer):
        return _get_atom_dim(atomic_encoder)
    return _get_bond_dim(atomic_encoder)

def _check_bond(bond: Any) -> None:
    if not isinstance(bond, Chem.Bond) and bond is not None:
        raise ValueError('bond needs to be either a `rdkit.Chem.Bond` or None')

def _check_atom(atom: Any) -> None:
    if not isinstance(atom, Chem.Atom):
        raise ValueError('atom needs to be a `rdkit.Chem.Atom`')
