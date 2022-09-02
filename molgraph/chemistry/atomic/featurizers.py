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
from warnings import warn

from molgraph.chemistry.atomic.features import AtomicFeature
from molgraph.chemistry.atomic.encoders import AtomicEncoder
from molgraph.chemistry.atomic.encoders import FloatEncoder
from molgraph.chemistry.atomic.encoders import OrdinalEncoder
from molgraph.chemistry.atomic.encoders import NominalEncoder



class AtomicFeaturizer:

    '''Atomic featurizer.

    Args:
        features (list[AtomicFeature]):
            List of atomic features.
        output_dtype (str, np.dtype):
            The output dtype.

    **Examples:**

    Atom featurizer:

    >>> atom_featurizer = molgraph.chemistry.AtomicFeaturizer([
    ...     molgraph.chemistry.features.Symbol(
    ...         allowable_set={'C', 'N'},
    ...         ordinal=False,
    ...         oov_size=1
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

    Bond featurizer:

    >>> bond_featurizer = molgraph.chemistry.AtomicFeaturizer([
    ...     molgraph.chemistry.features.BondType(
    ...         allowable_set={'SINGLE', 'DOUBLE'},
    ...         ordinal=False,
    ...         oov_size=1
    ...     ),
    ... ])
    >>> # Obtain a Bond
    >>> rdkit_mol = rdkit.Chem.MolFromSmiles('CC')
    >>> rdkit_bond = rdkit_mol.GetBondWithIdx(0)
    >>> # Encode Bond as a numerical vector
    >>> bond_featurizer(rdkit_bond)
    array([0., 0., 1.], dtype=float32)
    '''

    def __init__(
        self,
        features: List[AtomicFeature],
        output_dtype: str = 'float32'
    ) -> None:
        self._feature_type = _validate_features(features)
        self.features = _wrap_features(features)
        self.output_dtype = output_dtype
        self._ndim = _get_ndim(self, self._feature_type)

    def __call__(
        self,
        inputs: Union[
            List[Chem.Atom],
            List[Union[Chem.Bond, None]],
            Chem.Atom,
            Union[Chem.Bond, None],
        ]
    ) -> np.ndarray:
        '''Featurizes RDKit atom(s) or bond(s).

        Args:
            inputs (list, rdkit.Chem.Atom, rdkit.Chem.Bond):
                Either a single RDKit atom, a single RDKit bond, a list of RDKit
                atoms, or a list of RDKit bonds. If bonds do not exist for a
                given molecule, list of bonds will be an empty list. And if
                bond is a self loop, the bond will be represented as ``None``.

        Returns:
            np.ndarray: numerical encodings of atom(s) or bond(s).
        '''
        if isinstance(inputs, Chem.rdchem._ROAtomSeq):
            inputs = list(inputs)

        if not isinstance(inputs, (list, tuple, set, np.ndarray)):
            return np.concatenate([
                feature(inputs) for feature in self.features
            ]).astype(self.output_dtype)

        if self._feature_type == 'atom':
            return self._encode_atoms(inputs)
        return self._encode_bonds(inputs)

    def _encode_atoms(self, inputs: List[Chem.Atom]) -> np.ndarray:
        'Featurizes a list of atoms (rdkit.Chem.Atom).'
        atom_features = []
        for atom in inputs:
            # Make sure atom is a Chem.Atom
            _check_atom(atom)
            encoding = self(atom)
            atom_features.append(encoding)
        return np.asarray(atom_features, dtype=self.output_dtype)

    def _encode_bonds(
        self,
        inputs: List[Union[Chem.Bond, None]],
        self_loops: bool = False,
    ) -> np.ndarray:
        'Featurizes a list of bonds (rdkit.Chem.Bond or None).'

        ndim = self._ndim
        # Increase dim by 1 if self loops exist
        if self_loops:
            ndim += 1

        # If no bonds are supplied, return an "empty" array
        if not len(inputs):
            return np.zeros([0, ndim]).astype(self.output_dtype)

        bond_features = []
        for bond in inputs:
            # Make sure bond is either a Chem.Bond or None (self loop)
            _check_bond(bond)
            if bond is None:
                encoding = np.zeros(ndim, dtype=self.output_dtype)
                encoding[-1] = 1
            else:
                encoding = self(bond)
                pad_length = ndim - encoding.shape[0]
                if pad_length:
                    # If self loops are used, encoding is zero-padded by 1
                    encoding = np.pad(encoding, [(0, pad_length)])

            bond_features.append(encoding)

        return np.asarray(bond_features, dtype=self.output_dtype)

    def __repr__(self) -> str:
        return f'AtomicFeaturizer(features={self.features})'


class AtomFeaturizer(AtomicFeaturizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warn(f'{self.__class__.__name__} will be deprecated in the near future',
            DeprecationWarning, stacklevel=2)


class BondFeaturizer(AtomicFeaturizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warn(f'{self.__class__.__name__} will be deprecated in the near future',
            DeprecationWarning, stacklevel=2)


def _validate_features(features: List[AtomicFeature]):
    dummy_mol = Chem.MolFromSmiles('CC')
    dummy_atom = dummy_mol.GetAtomWithIdx(0)
    dummy_bond = dummy_mol.GetBondWithIdx(0)
    try:
        # Check if features are atom features
        _ = [f(dummy_atom) for f in features]
        feature_type = 'atom'
    except:
        try:
            # Check if features are bond features
            _ = [f(dummy_bond) for f in features]
            feature_type = 'bond'
        except:
            feature_type = None

    if feature_type is None:
        raise ValueError('Invalid `features`.')

    return feature_type

def _wrap_features(
    features: List[AtomicFeature],
) -> List[AtomicEncoder]:
    wrapped_features = []
    for f in features:
        if not hasattr(f, 'allowable_set'):
            wrapped_features.append(FloatEncoder(f))
        elif getattr(f, 'ordinal', False):
            wrapped_features.append(OrdinalEncoder(f))
        else:
            wrapped_features.append(NominalEncoder(f))
    return wrapped_features

def _get_ndim(atomic_encoder: AtomicFeaturizer, feature_type: str):
    dummy_mol = Chem.MolFromSmiles('CC')
    if feature_type == 'atom':
        return len(atomic_encoder(dummy_mol.GetAtomWithIdx(0)))
    return len(atomic_encoder(dummy_mol.GetBondWithIdx(0)))

def _check_bond(bond: Any) -> None:
    if not isinstance(bond, Chem.Bond) and bond is not None:
        raise ValueError('bond needs to be either a `rdkit.Chem.Bond` or None')

def _check_atom(atom: Any) -> None:
    if not isinstance(atom, Chem.Atom):
        raise ValueError('atom needs to be a `rdkit.Chem.Atom`')
