import numpy as np
from dataclasses import dataclass
from dataclasses import field
from typing import Union
from typing import Tuple
from typing import List
from typing import Any
from typing import Dict
from rdkit import Chem
from warnings import warn

from molgraph.chemistry.atomic.features import AtomicFeature
from molgraph.chemistry.atomic.encoders import TokenEncoder



class AtomicTokenizer:

    '''Atomic tokenizer.

    Args:
        features (list[AtomicFeature]):
            List of atomic (atom or bond) features.

    **Example:**

    >>> atom_tokenizer = molgraph.chemistry.AtomicTokenizer([
    ...     molgraph.chemistry.features.Symbol(
    ...         allowable_set={'C', 'N'},           # irrelevant
    ...         ordinal=False,                      # irrelevant
    ...         oov_size=0                          # irrelevant
    ...     ),
    ...     molgraph.chemistry.features.Hybridization(
    ...         allowable_set={'SP', 'SP2', 'SP3'},
    ...     )
    ... ])
    >>> # Obtain an Atom
    >>> rdkit_mol = rdkit.Chem.MolFromSmiles('CC')
    >>> rdkit_atom = rdkit_mol.GetAtomWithIdx(0)
    >>> # Encode Atom as a token
    >>> atom_tokenizer(rdkit_atom)
    array('Sym:C|Hyb:SP3', dtype='<U13')
    '''

    def __init__(self, features):
        self._feature_type = _validate_features(features)
        self.features = _wrap_features(features)

    def __call__(
        self,
        inputs: Union[
            List[Chem.Atom],
            List[Union[Chem.Bond, None]],
            Chem.Atom,
            Union[Chem.Bond, None],
        ]
    ) -> np.ndarray:
        '''Tokenizes RDKit atom(s) or bond(s).

        Args:
            inputs (list, rdkit.Chem.Atom, rdkit.Chem.Bond):
                Either a single RDKit atom, a single RDKit bond, a list of RDKit
                atoms, or a list of RDKit bonds. If bonds do not exist for a
                given molecule, list of bonds will be an empty list. And if
                bond is a self loop, the bond will be represented as ``None``.

        Returns:
            np.ndarray: Token encoding of atom(s) or bond(s).
        '''
        if isinstance(inputs, Chem.rdchem._ROAtomSeq):
            inputs = list(inputs)

        if not isinstance(inputs, (list, tuple, set, np.ndarray)):
            return np.array(
                '|'.join([feature(inputs) for feature in self.features]),
                dtype=str)

        if self._feature_type == 'atom':
            return self._encode_atoms(inputs)
        return self._encode_bonds(inputs)

    def _encode_atoms(self, inputs: List[Chem.Atom]) -> np.ndarray:
        'Tokenizes a list of RDKit atoms (rdkit.Chem.Atom).'
        atom_tokens = []
        for atom in inputs:
            _check_atom(atom)
            encoding = self(atom)
            atom_tokens.append(encoding)
        return np.asarray(atom_tokens)

    def _encode_bonds(self, inputs: List[Chem.Bond]) -> np.ndarray:
        'Tokenizes a list of RDKit bonds (rdkit.Chem.Bond or None).'

        # If no bonds are supplied, return an "empty" array
        if not len(inputs):
            return np.zeros([0]).astype(str)

        bond_tokens = []
        for bond in inputs:
            _check_bond(bond)
            if bond is None:
                bond_tokens.append(np.array(['[SELF_LOOP]']))
            else:
                bond_tokens.append(self(bond))
        return np.asarray(bond_tokens)

    def __repr__(self) -> str:
        return f'AtomicTokenizer(features={self.features})'


class AtomTokenizer(AtomicTokenizer):
    def __init__(self, features):
        super().__init__(features)
        warn(f'{self.__class__.__name__} will be deprecated in the near future',
            DeprecationWarning, stacklevel=2)

class BondTokenizer(AtomicTokenizer):
    def __init__(self, features):
        super().__init__(features)
        warn(f'{self.__class__.__name__} will be deprecated in the near future',
            DeprecationWarning, stacklevel=2)

def _validate_features(features: List[AtomicFeature]):
    dummy_mol = Chem.MolFromSmiles('CC')
    dummy_atom = dummy_mol.GetAtomWithIdx(0)
    dummy_bond = dummy_mol.GetBondWithIdx(0)
    try:
        # Check if all features are atom features
        _ = [f(dummy_atom) for f in features]
        feature_type = 'atom'
    except:
        try:
            # Check if all features are bond features
            _ = [f(dummy_bond) for f in features]
            feature_type = 'bond'
        except:
            feature_type = None

    if feature_type is None:
        raise ValueError('Invalid `features`.')

    return feature_type

def _wrap_features(
    features: List[AtomicFeature],
) -> List[TokenEncoder]:
    return [TokenEncoder(f) for f in features]

def _check_bond(bond: Any) -> None:
    if not isinstance(bond, Chem.Bond) and bond is not None:
        raise ValueError('bond needs to be either a `Chem.Bond` or None')

def _check_atom(atom: Any) -> None:
    if not isinstance(atom, Chem.Atom):
        raise ValueError('atom needs to be a `Chem.Atom`')
