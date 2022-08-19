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
from typing import NewType


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

    '''Base class for atom and bond features.

    Create a custom feature by subclassnig this class. Make sure to define
    a ``call()`` method that computes from a ``rdkit.Chem.Atom`` a feature.
    The feature should be a ``str``, ``float`` or ``int``. For example:

    >>> class MyAtomMassFeature(molgraph.chemistry.AtomicFeature):
    ...     def call(self, atom: rdkit.Chem.Atom) -> float:
    ...         return atom.GetMass()
    >>> # Instantiate feature
    >>> my_feature = MyAtomMassFeature()
    >>> # Compute feature of atom
    >>> atom = rdkit.Chem.MolFromSmiles('CC').GetAtomWithIdx(0)
    >>> my_feature(atom)
    12.011

    Incorporate custom feature into e.g. ``AtomFeaturizer``:

    >>> class MySymbolFeature(molgraph.chemistry.AtomicFeature):
    ...     def call(self, atom: rdkit.Chem.Atom) -> str:
    ...         return atom.GetSymbol()
    >>> # Obtain an atom
    >>> atom = rdkit.Chem.MolFromSmiles('COO').GetAtomWithIdx(1)
    >>> # Build atom featurizer
    >>> atom_featurizer = molgraph.chemistry.AtomFeaturizer([
    ...     MySymbolFeature(allowable_set={'C', 'O'}, oov_size=1),
    ...     molgraph.chemistry.features.HydrogenAcceptor()
    ... ])
    >>> # Compute numerical encoding of atom via featurizer
    >>> atom_featurizer(atom)
    array([0., 0., 1., 1.], dtype=float32)

    Args:
        allowable_set (set, list, tuple):
            A set of features which should be considered.
        ordinal (bool):
            Whether the feature should be encoded as an ordinal vector. Will
            only take effect when the feature is used in a featurizer. If
            feature is to be tokenized, this parameter will be ignored. Default
            to False.
        oov_size (int):
            The number of slots for an OOV feature. If feature is to be
            tokenized, this parameter will be ignored. Default to 0.
    '''

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
        'Returns a raw feature from an RDKit atom or bond'
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
        'Transforms an RDKit atom or bond to a feature.'
        pass


class AtomicFeatureFactory:

    '''An atomic feature factory.

    Two feature factories exist by default, both from which features can be
    obtained: ``atom_features`` and ``bond_features``:

    >>> molgraph.chemistry.atom_features.registered_features
    ['symbol',
     'hybridization',
     'cip_code',
     'chiral_center',
     'formal_charge',
     'total_num_hs',
     'total_valence',
     'num_radical_electrons',
     'degree',
     'aromatic',
     'hetero',
     'hydrogen_donor',
     'hydrogen_acceptor',
     'ring_size',
     'ring',
     'crippen_log_p_contribution',
     'crippen_molar_refractivity_contribution',
     'tpsa_contribution',
     'labute_asa_contribution',
     'gasteiger_charge',
     'atom_mass']
    >>> molgraph.chemistry.atom_features.get('cip_code')
    CIPCode(allowable_set={'R', 'S', None}, ordinal=False, oov_size=0)

    '''

    def __init__(self, feature_type):
        self._feature_type = feature_type
        self._features = {}


    def register_feature(
        self,
        feature: AtomicFeature,
        name: str
    ) -> None:
        '''Registers a derived class of ``AtomicFeature``.

        **Example:**

        >>> class MyAtomMassFeature(molgraph.chemistry.AtomicFeature):
        ...     def call(self, atom: rdkit.Chem.Atom) -> float:
        ...         return atom.GetMass()
        >>> molgraph.chemistry.atom_features.register_feature(
        ...     MyAtomMassFeature, 'atom_mass')
        >>> # Obtain my feature
        >>> molgraph.chemistry.atom_features.get('atom_mass')
        MyAtomMassFeature()

        '''
        self._features[name] = feature

    def register(
        self,
        name: str
    ) -> Callable[[AtomicFeature], AtomicFeature]:
        '''Registers a derived class of ``AtomicFeature`` via decoration.

        **Example:**

        >>> @molgraph.chemistry.atom_features.register('atom_mass')
        >>> class MyAtomMassFeature(molgraph.chemistry.AtomicFeature):
        ...     def call(self, atom: rdkit.Chem.Atom) -> float:
        ...         return atom.GetMass()
        >>> # Obtain my feature
        >>> molgraph.chemistry.atom_features.get('atom_mass')
        MyAtomMassFeature()

        '''
        def wrapper(feature: AtomicFeature) -> AtomicFeature:
            self._inject_docstring(feature)
            self.register_feature(feature, name)
            return feature
        return wrapper

    def _inject_docstring(self, feature: AtomicFeature) -> None:
        s = self._feature_type.capitalize()
        feature.__doc__ = f'{s} feature.'
        feature.call.__doc__ = f'''Transforms a ``rdkit.Chem.{s}`` to a feature.

        Args:
            {s.lower()} (rdkit.Chem.{s}):
                The input to be transformed to a feature.
        '''

    def get(self, name, *args, **kwargs):
        'Get feature by name (see ``registered_features``).'
        return self._features[name](*args, **kwargs)

    @property
    def registered_features(self) -> List[str]:
        'Lists all registered features.'
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


atom_features = AtomicFeatureFactory('atom')
bond_features = AtomicFeatureFactory('bond')


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
