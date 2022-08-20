import numpy as np
from dataclasses import dataclass
from dataclasses import field
from typing import Union
from typing import Tuple
from typing import List
from typing import Any
from typing import Dict
from rdkit import Chem

from molgraph.chemistry.atomic.features import AtomicFeature
from molgraph.chemistry.atomic.encoders import TokenEncoder



@dataclass
class AtomicTokenizer:

    features: List[AtomicFeature]

    def __post_init__(self) -> None:
        self.features = [TokenEncoder(f) for f in self.features]

    def __call__(
        self,
        inputs: Union[Chem.Atom, Chem.Bond]
    ) -> Union[np.ndarray, Dict[str, int]]:
        '''Tokenizes a single rdkit Atom or Bond.

        Args:
            inputs (rdkit.Chem.Atom, rdkit.Chem.Bond):
                <placeholder>

        Returns:
            np.ndarray: Token encodings of an atom or a bond.
        '''
        return np.array(
            '|'.join([feature(inputs) for feature in self.features]), dtype=str)


class AtomTokenizer(AtomicTokenizer):

    '''Atom tokenizer.

    Args:
        features (list[AtomicFeature]):
            <placeholder>
        dtype (str, np.dtype):
            <placeholder>

    **Example:**

    >>> atom_tokenizer  = molgraph.chemistry.AtomTokenizer([
    ...     molgraph.chemistry.features.Symbol(
    ...         allowable_set={'C', 'N'},           #   relevant
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

    def encode_atoms(self, inputs: List[Chem.Atom]) -> np.ndarray:
        '''Tokenizes a list of RDKit atoms (rdkit.Chem.Atom).

        Args:
            inputs (list[rdkit.Chem.Atom]):
                <placeholder>

        Returns:
            np.ndarray: Token encodings of multiple atoms.
        '''
        atom_tokens = []
        for atom in inputs:
            _check_atom(atom)
            atom_tokens.append(self(atom))
        return np.asarray(atom_tokens)


class BondTokenizer(AtomicTokenizer):

    '''Bond tokenizer.

    Args:
        features (list[AtomicFeature]):
            <placeholder>
        dtype (str, np.dtype):
            <placeholder>

    **Example:**

    >>> bond_tokenizer = molgraph.chemistry.BondTokenizer([
    ...     molgraph.chemistry.features.BondType(
    ...         allowable_set={'SINGLE', 'DOUBLE'}, #   relevant
    ...         ordinal=False,                      # irrelevant
    ...         oov_size=0                          # irrelevant
    ...     )
    ... ])
    >>> # Obtain a Bond
    >>> rdkit_mol = rdkit.Chem.MolFromSmiles('CC')
    >>> rdkit_bond = rdkit_mol.GetBondWithIdx(0)
    >>> # Encode Bond as a token
    >>> bond_tokenizer(rdkit_bond)
    array('BonTyp:SINGLE', dtype='<U13')
    '''

    def encode_bonds(self, inputs: List[Chem.Bond]) -> np.ndarray:
        '''Tokenizes a list of RDKit bonds (rdkit.Chem.Bond).

        Args:
            inputs (list[rdkit.Chem.Bond]):
                <placeholder>

        Returns:
            np.ndarray: Token encodings of multiple bonds.
        '''
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


def _check_bond(bond: Any) -> None:
    if not isinstance(bond, Chem.Bond) and bond is not None:
        raise ValueError('bond needs to be either a `Chem.Bond` or None')

def _check_atom(atom: Any) -> None:
    if not isinstance(atom, Chem.Atom):
        raise ValueError('atom needs to be a `Chem.Atom`')
