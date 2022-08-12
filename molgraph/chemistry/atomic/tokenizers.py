import numpy as np
from dataclasses import dataclass
from dataclasses import field
from typing import Union
from typing import Tuple
from typing import List
from typing import Any
from typing import Dict
from rdkit import Chem

from molgraph.chemistry.atomic import features
from molgraph.chemistry.atomic import encodings



@dataclass
class AtomicTokenizer:

    features: List[features.AtomicFeature]

    def __post_init__(self) -> None:
        self._feature_names = []
        for i, feature in enumerate(self.features):
            self._feature_names.append(feature.name)
            self.features[i] = encodings.TokenEncoding(feature)

    def __call__(
        self,
        x: Union[Chem.Atom, Chem.Bond]
    ) -> Union[np.ndarray, Dict[str, int]]:
        """Converts a rdkit Atom or Bond to an encoding of the Atom or Bond.
        """
        return np.array(
            '|'.join([feature(x) for feature in self.features]), dtype=str)

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names


class AtomTokenizer(AtomicTokenizer):

    def encode_atoms(self, atoms: List[Chem.Atom]) -> np.ndarray:
        atom_tokens = []
        for atom in atoms:
            _check_atom(atom)
            atom_tokens.append(self(atom))
        return np.asarray(atom_tokens)


class BondTokenizer(AtomicTokenizer):

    def encode_bonds(self, bonds: List[Chem.Bond]) -> np.ndarray:
        if not len(bonds):
            return np.zeros([0]).astype(str)
        bond_tokens = []
        for bond in bonds:
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
