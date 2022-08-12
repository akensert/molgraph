import numpy as np
from rdkit import Chem
from rdkit.Chem import Lipinski
from rdkit.Chem import Crippen
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdPartialCharges
from dataclasses import dataclass
from dataclasses import field
from abc import ABC
from abc import abstractmethod
from typing import Union
from typing import Any
from typing import Callable
from typing import List
from typing import Sequence
from typing import Optional


_defaults = {
    'Symbol': {
        'H',  'He', 'Li', 'Be', 'B',  'C',  'N',  'O',  'F',  'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P',  'S',  'Cl', 'Ar', 'K',  'Ca',
        'Sc', 'Ti', 'V',  'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y',  'Zr',
        'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
        'Sb', 'Te', 'I',  'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
        'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
        'Lu', 'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
        'Pa', 'U',  'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
        'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
        'Rg', 'Cn'
    },
    'Hybridization': {'S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'UNSPECIFIED'},
    'CIPCode': {'R', 'S', None},
    'FormalCharge': {-3, -2, -1, 0, 1, 2, 3, 4},
    'TotalNumHs': {0, 1, 2, 3, 4},
    'TotalValence': {0, 1, 2, 3, 4, 5, 6, 7, 8},
    'NumRadicalElectrons': {0, 1, 2, 3},
    'Degree': {0, 1, 2, 3, 4, 5, 6, 7, 8},
    'RingSize': {0, 3, 4, 5, 6, 7, 8},
    'BondType': {'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'},
    'Stereo': {'STEREOE', 'STEREOZ', 'STEREOANY', 'STEREONONE'},
}


class AtomicFeature(ABC):

    """Base class for features, both atom and bond features. Create a custom
    feature by subclassnig this base class with a mandatory `call` method.
    If a `set` of features can be generated with the subclass, define an
    `allowable_set`. `ordinal` and `oov_size` are ignored if `allowable_set`
    is not supplied, or if feature is to be tokenized.
    """

    def __init__(
        self,
        allowable_set: Optional[Sequence[Any]] = None,
        ordinal: bool = False,
        oov_size: int = 0,
    ) -> None:
        'Initializes attributes'
        self.allowable_set = allowable_set
        self.ordinal = ordinal
        self.oov_size = oov_size

        if self.allowable_set is None:
            self.allowable_set = _defaults.get(self.name, None)
            if self.allowable_set is None:
                self.ordinal = None
                self.oov_size = None
        if self.ordinal:
            self.oov_size = None

    def __call__(self, atom_or_bond: Union[Chem.Atom, Chem.Bond]) -> str:
        'Returns a raw feature from RDKit atom or bond'
        return self.call(atom_or_bond)

    def __repr__(self):
        fields = []
        if self.allowable_set is not None:
            fields.append(f"allowable_set={self.allowable_set}")
        if self.ordinal is not None:
            fields.append(f"ordinal={self.ordinal}")
        if self.oov_size is not None:
            fields.append(f"oov_size={self.oov_size}")
        return self.__class__.__name__ + '(' + ', '.join(fields) + ')'

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def call(self, x: Union[Chem.Atom, Chem.Bond]) -> Any:
        pass


class AtomicFeatureFactory:

    def __init__(self):
        self._features = {}

    def register_feature(self, feature: AtomicFeature, name: str) -> None:
        self._features[name] = feature

    def register(self, name: str) -> Callable[[AtomicFeature], AtomicFeature]:
        '''Decorator for a feature (AtomicFeature)'''
        def wrapper(feature: AtomicFeature) -> AtomicFeature:
            self.register_feature(feature, name)
            return feature
        return wrapper

    def get(self, name, *args, **kwargs):
        return self._features[name](*args, **kwargs)

    @property
    def registered_features(self) -> List[str]:
        return list(self._features.keys())

    def __repr__(self):
        class_name = self.__class__.__name__
        return (
            class_name + '(registered_features=' +
            f'{list(self._features.keys())}' + ')'
        )

    def unpack(
        self,
        exclude: Optional[Sequence[str]] = None
    ) -> List[AtomicFeature]:
        if exclude is None or isinstance(exclude, str):
            exclude = [exclude]
        return [v() for (k, v) in self._features.items() if k not in exclude]


atom_features = AtomicFeatureFactory()
bond_features = AtomicFeatureFactory()


@atom_features.register(name='symbol')
class Symbol(AtomicFeature):
    def call(self, atom: Chem.Atom) -> str:
        return atom.GetSymbol()


@atom_features.register(name='hybridization')
class Hybridization(AtomicFeature):
    def call(self, atom: Chem.Atom) -> str:
        return atom.GetHybridization().name


@atom_features.register(name='cip_code')
class CIPCode(AtomicFeature):
    def call(self, atom: Chem.Atom) -> Union[None, str]:
        if atom.HasProp("_CIPCode"):
            return atom.GetProp("_CIPCode")
        return None


@atom_features.register(name='chiral_center')
class ChiralCenter(AtomicFeature):
    def call(self, atom: Chem.Atom) -> bool:
        return atom.HasProp("_ChiralityPossible")


@atom_features.register(name='formal_charge')
class FormalCharge(AtomicFeature):
    def call(self, atom: Chem.Atom) -> int:
        return atom.GetFormalCharge()


@atom_features.register(name='total_num_hs')
class TotalNumHs(AtomicFeature):
    def call(self, atom: Chem.Atom) -> int:
        return atom.GetTotalNumHs()


@atom_features.register(name='total_valence')
class TotalValence(AtomicFeature):
    def call(self, atom: Chem.Atom) -> int:
        return atom.GetTotalValence()


@atom_features.register(name='num_radical_electrons')
class NumRadicalElectrons(AtomicFeature):
    def call(self, atom: Chem.Atom) -> int:
        return atom.GetNumRadicalElectrons()


@atom_features.register(name='degree')
class Degree(AtomicFeature):
    def call(self, atom: Chem.Atom) -> int:
        return atom.GetDegree()


@atom_features.register(name='aromatic')
class Aromatic(AtomicFeature):
    def call(self, atom: Chem.Atom) -> bool:
        return atom.GetIsAromatic()


@atom_features.register(name='hetero')
class Hetero(AtomicFeature):
    def call(self, atom: Chem.Atom) -> bool:
        mol = atom.GetOwningMol()
        return atom.GetIdx() in [i[0] for i in Lipinski._Heteroatoms(mol)]


@atom_features.register(name='hydrogen_donor')
class HydrogenDonor(AtomicFeature):
    def call(self, atom: Chem.Atom) -> bool:
        mol = atom.GetOwningMol()
        return atom.GetIdx() in [i[0] for i in Lipinski._HDonors(mol)]


@atom_features.register(name='hydrogen_acceptor')
class HydrogenAcceptor(AtomicFeature):
    def call(self, atom: Chem.Atom) -> bool:
        mol = atom.GetOwningMol()
        return atom.GetIdx() in [i[0] for i in Lipinski._HAcceptors(mol)]


@atom_features.register(name='ring_size')
class RingSize(AtomicFeature):
    def call(self, atom: Chem.Atom) -> int:
        size = 0
        if atom.IsInRing():
            while not atom.IsInRingSize(size):
                size += 1
        return size


@atom_features.register(name='ring')
class Ring(AtomicFeature):
    def call(self, atom: Chem.Atom) -> bool:
        return atom.IsInRing()


@atom_features.register(name='crippen_log_p_contribution')
class CrippenLogPContribution(AtomicFeature):
    def call(self, atom: Chem.Atom) -> float:
        mol = atom.GetOwningMol()
        val = Crippen._GetAtomContribs(mol)[atom.GetIdx()][0]
        return val if val is not None else 0.0


@atom_features.register(name='crippen_molar_refractivity_contribution')
class CrippenMolarRefractivityContribution(AtomicFeature):
    def call(self, atom: Chem.Atom) -> float:
        mol = atom.GetOwningMol()
        val = Crippen._GetAtomContribs(mol)[atom.GetIdx()][1]
        return val if val is not None else 0.0


@atom_features.register(name='tpsa_contribution')
class TPSAContribution(AtomicFeature):
    def call(self, atom: Chem.Atom) -> float:
        mol = atom.GetOwningMol()
        val = rdMolDescriptors._CalcTPSAContribs(mol)[atom.GetIdx()]
        return val if val is not None else 0.0


@atom_features.register(name='labute_asa_contribution')
class LabuteASAContribution(AtomicFeature):
    def call(self, atom: Chem.Atom) -> float:
        mol = atom.GetOwningMol()
        val = rdMolDescriptors._CalcLabuteASAContribs(mol)[0][atom.GetIdx()]
        return val if val is not None else 0.0


@atom_features.register(name='gasteiger_charge')
class GasteigerCharge(AtomicFeature):
    def call(self, atom: Chem.Atom) -> float:
        mol = atom.GetOwningMol()
        rdPartialCharges.ComputeGasteigerCharges(mol)
        val = atom.GetDoubleProp('_GasteigerCharge')
        return val if val is not None else 0.0


@bond_features.register(name='bond_type')
class BondType(AtomicFeature):
    def call(self, bond: Chem.Bond) -> str:
        return bond.GetBondType().name


@bond_features.register(name='conjugated')
class Conjugated(AtomicFeature):
    def call(self, bond: Chem.Bond) -> bool:
        return bond.GetIsConjugated()


@bond_features.register(name='rotatable')
class Rotatable(AtomicFeature):
    def call(self, bond: Chem.Bond) -> bool:
        mol = bond.GetOwningMol()
        atom_indices = tuple(
            sorted([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]))
        return atom_indices in Lipinski._RotatableBonds(mol)


@bond_features.register(name='stereo')
class Stereo(AtomicFeature):
    def call(self, bond: Chem.Bond) -> str:
        return bond.GetStereo().name
