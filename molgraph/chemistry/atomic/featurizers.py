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

from molgraph.chemistry.atomic import features
from molgraph.chemistry.atomic import encodings



@dataclass
class AtomicFeaturizer:

    features: List[features.AtomicFeature]
    dtype: Union[str, np.dtype] = 'float32'
    ndim: int = field(init=False)

    def __post_init__(self) -> None:
        self._feature_names = []
        for i, feature in enumerate(self.features):
            self._feature_names.append(feature.name)
            if not getattr(feature, 'allowable_set', None):
                self.features[i] = encodings.FloatEncoding(feature)
            elif not feature.ordinal:
                self.features[i] = encodings.NominalEncoding(feature)
            else:
                self.features[i] = encodings.OrdinalEncoding(feature)

        self.ndim = _get_atom_or_bond_dim(self)

    def __call__(self, x: Union[Chem.Atom, Chem.Bond]) -> np.ndarray:
        """Encodes a single rdkit Atom or Bond."""
        return np.concatenate([feature(x) for feature in self.features])

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names


class AtomFeaturizer(AtomicFeaturizer):

    def encode_atoms(self, atoms: List[Chem.Atom]) -> np.ndarray:
        atom_features = []
        for atom in atoms:
            _check_atom(atom)
            atom_features.append(self(atom))
        return np.asarray(atom_features, dtype=self.dtype)


class BondFeaturizer(AtomicFeaturizer):

    def encode_bonds(
        self,
        bonds: List[Chem.Bond],
        self_loops: bool = False,
    ) -> np.ndarray:

        if self_loops:
            self.ndim += 1

        if not len(bonds):
            return np.zeros([0, self.ndim]).astype(self.dtype)

        bond_features = []
        for bond in bonds:
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
        raise ValueError('bond needs to be either a `Chem.Bond` or None')

def _check_atom(atom: Any) -> None:
    if not isinstance(atom, Chem.Atom):
        raise ValueError('atom needs to be a `Chem.Atom`')
