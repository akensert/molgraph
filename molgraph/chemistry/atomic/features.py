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

    '''Atomic feature.

    Defines an atom or bond feature, which can subsequently be passed to
    an ``chemistry.AtomicFeaturizer`` or ``chemistry.AtomicTokenizer`` to be
    featurized or tokenized respectively.

    Arguments (see below) will only take effect when ``AtomicFeature`` is
    wrapped in an ``AtomicEncoder``, which occurs automatically inside
    ``chemistry.AtomicFeaturizer`` or ``chemistry.AtomicTokenizer``.

    Args:
        allowable_set (set, list, tuple, None):
            A set of features that will be considered. If None, default
            set will be used, if it exists. Default to None.
        ordinal (bool):
            Whether to encode the feature as an ordinal vector. Only relevant
            if ``allowable_set`` exists and feature is passed to an
            ``AtomicFeatuizer``. Default to False.
        oov_size (int):
            The number of bins alloted to OOV features. Only relevant if
            ``allowable_set`` exists. Default to 0.

    **Examples:**

    Pass features to ``chemistry.AtomicFeaturizer`` to encode atom(s)

    >>> # Obtain RDKit atoms, via the RDKit API
    >>> atoms = rdkit.Chem.MolFromSmiles('COO').GetAtoms()
    >>> # Build an atom featurizer, from AtomicFeaturizer
    >>> atom_featurizer = molgraph.chemistry.AtomicFeaturizer([
    ...     molgraph.chemistry.features.Symbol(
    ...         allowable_set={'C', 'O'}, oov_size=1),     # specify param
    ...     molgraph.chemistry.features.HydrogenAcceptor() # use default param
    ... ])
    >>> # Compute numerical encoding of atoms. (OOV bin is prepended.)
    >>> atom_featurizer(atoms)
    array([[0., 1., 0., 0.],
           [0., 0., 1., 1.],
           [0., 0., 1., 1.]], dtype=float32)


    Create custom features by subclassing ``chemistry.AtomicFeature``

    >>> class MySymbolFeature(molgraph.chemistry.AtomicFeature):
    ...     def __call__(self, atom: rdkit.Chem.Atom) -> str:
    ...         return atom.GetSymbol()
    >>> # Obtain RDKit atoms, via the RDKit API
    >>> atoms = rdkit.Chem.MolFromSmiles('COO').GetAtoms()
    >>> # Build an atom featurizer, from AtomicFeaturizer
    >>> atom_featurizer = molgraph.chemistry.AtomicFeaturizer([
    ...     MySymbolFeature(allowable_set={'C', 'O'}, oov_size=1),
    ...     molgraph.chemistry.features.HydrogenAcceptor()
    ... ])
    >>> # Compute numerical encoding of atoms. (OOV bin is prepended.)
    >>> atom_featurizer(atoms)
    array([[0., 1., 0., 0.],
           [0., 0., 1., 1.],
           [0., 0., 1., 1.]], dtype=float32)


    Utilize feature factories ``chemistry.atom_features`` and
    ``chemistry.bond_features``

    >>> molgraph.chemistry.bond_features.registered_features
    ['bond_type',
    'conjugated',
    'rotatable',
    'stereo']
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
     'gasteiger_charge']
    >>> molgraph.chemistry.atom_features.get('cip_code')
    CIPCode(allowable_set={'R', 'S', None}, ordinal=False, oov_size=0)
    >>> molgraph.chemistry.atom_features.get('cip_code', ordinal=True)
    CIPCode(allowable_set={'R', 'S', None}, ordinal=True, oov_size=0)
    '''

    def __init__(
        self,
        allowable_set: Optional[Sequence[Any]] = None,
        ordinal: str = False,
        oov_size: int = 0,
    ) -> None:

        if allowable_set is None:
            allowable_set = _defaults.get(self.name, None)

        if allowable_set is not None:
            self.allowable_set = list(allowable_set)
            self.ordinal = ordinal
            if ordinal:
                self.oov_size = 0
            else:
                self.oov_size = oov_size

    @abstractmethod
    def __call__(self, inputs: Union[Chem.Atom, Chem.Bond]) -> str:
        'Obtain feature for RDKit atom(s) or bond(s).'
        pass

    def __repr__(self):
        fields = [f'{k}={v}' for k, v in self.__dict__.items()]
        return self.name + '(' + ', '.join(fields) + ')'

    @property
    def name(self) -> str:
        return self.__class__.__name__


class AtomicFeatureFactory:

    'A factory for atomic features.'

    def __init__(self, feature_type):
        self._feature_type = feature_type
        self._features = {}

    def register(
        self,
        name: str
    ) -> Callable[[AtomicFeature], AtomicFeature]:
        '''Registers a derived class of ``AtomicFeature`` via decoration.

        **Example:**

        >>> @molgraph.chemistry.atom_features.register('atom_mass')
        >>> class AtomMass(molgraph.chemistry.AtomicFeature):
        ...     def __call__(self, atom: rdkit.Chem.Atom) -> float:
        ...         return atom.GetMass()
        >>> # Obtain atom mass feature
        >>> molgraph.chemistry.atom_features.get('atom_mass')
        AtomMass()
        '''
        def wrapper(feature: AtomicFeature) -> AtomicFeature:
            self._add_docs(feature, self._feature_type)
            self._features[name] = feature
            return feature
        return wrapper

    def get(self, name, *args, **kwargs):
        'Get feature by name (see ``registered_features``).'
        return self._features[name](*args, **kwargs)

    def get_all(
        self,
        exclude: Optional[Sequence[str]] = None
    ) -> List[AtomicFeature]:
        '''Instantiates and returns all available features (see
        ``registered features``).
        '''
        if exclude is None or isinstance(exclude, str):
            exclude = [exclude]
        return [v() for (k, v) in self._features.items() if k not in exclude]

    @property
    def registered_features(self) -> List[str]:
        'Lists all registered features.'
        return list(self._features.keys())

    @staticmethod
    def _add_docs(feature: AtomicFeature, string: str) -> None:
        feature.__doc__ = (
            f'''{string.capitalize()} feature.'''
        )
        feature.__call__.__doc__ = (
            f'''Transforms an ``rdkit.Chem.{string.capitalize()}`` to a feature.

            Args:
                {string.lower()} (rdkit.Chem.{string.capitalize()}):
                    The input to be transformed to a feature.
            '''
        )

    def __repr__(self):
        class_name = self.__class__.__name__
        return class_name + f'(registered_features={self.registered_features})'


atom_features = AtomicFeatureFactory('atom')
bond_features = AtomicFeatureFactory('bond')


@atom_features.register(name='symbol')
class Symbol(AtomicFeature):
    def __call__(self, atom: Chem.Atom) -> str:
        return atom.GetSymbol()


@atom_features.register(name='hybridization')
class Hybridization(AtomicFeature):
    def __call__(self, atom: Chem.Atom) -> str:
        return atom.GetHybridization().name


@atom_features.register(name='cip_code')
class CIPCode(AtomicFeature):
    def __call__(self, atom: Chem.Atom) -> Union[None, str]:
        if atom.HasProp("_CIPCode"):
            return atom.GetProp("_CIPCode")
        return None


@atom_features.register(name='chiral_center')
class ChiralCenter(AtomicFeature):
    def __call__(self, atom: Chem.Atom) -> bool:
        return atom.HasProp("_ChiralityPossible")


@atom_features.register(name='formal_charge')
class FormalCharge(AtomicFeature):
    def __call__(self, atom: Chem.Atom) -> int:
        return atom.GetFormalCharge()


@atom_features.register(name='total_num_hs')
class TotalNumHs(AtomicFeature):
    def __call__(self, atom: Chem.Atom) -> int:
        return atom.GetTotalNumHs()


@atom_features.register(name='total_valence')
class TotalValence(AtomicFeature):
    def __call__(self, atom: Chem.Atom) -> int:
        return atom.GetTotalValence()


@atom_features.register(name='num_radical_electrons')
class NumRadicalElectrons(AtomicFeature):
    def __call__(self, atom: Chem.Atom) -> int:
        return atom.GetNumRadicalElectrons()


@atom_features.register(name='degree')
class Degree(AtomicFeature):
    def __call__(self, atom: Chem.Atom) -> int:
        return atom.GetDegree()


@atom_features.register(name='aromatic')
class Aromatic(AtomicFeature):
    def __call__(self, atom: Chem.Atom) -> bool:
        return atom.GetIsAromatic()


@atom_features.register(name='hetero')
class Hetero(AtomicFeature):
    def __call__(self, atom: Chem.Atom) -> bool:
        mol = atom.GetOwningMol()
        return atom.GetIdx() in [i[0] for i in Lipinski._Heteroatoms(mol)]


@atom_features.register(name='hydrogen_donor')
class HydrogenDonor(AtomicFeature):
    def __call__(self, atom: Chem.Atom) -> bool:
        mol = atom.GetOwningMol()
        return atom.GetIdx() in [i[0] for i in Lipinski._HDonors(mol)]


@atom_features.register(name='hydrogen_acceptor')
class HydrogenAcceptor(AtomicFeature):
    def __call__(self, atom: Chem.Atom) -> bool:
        mol = atom.GetOwningMol()
        return atom.GetIdx() in [i[0] for i in Lipinski._HAcceptors(mol)]


@atom_features.register(name='ring_size')
class RingSize(AtomicFeature):
    def __call__(self, atom: Chem.Atom) -> int:
        size = 0
        if atom.IsInRing():
            while not atom.IsInRingSize(size):
                size += 1
        return size


@atom_features.register(name='ring')
class Ring(AtomicFeature):
    def __call__(self, atom: Chem.Atom) -> bool:
        return atom.IsInRing()


@atom_features.register(name='crippen_log_p_contribution')
class CrippenLogPContribution(AtomicFeature):
    def __call__(self, atom: Chem.Atom) -> float:
        mol = atom.GetOwningMol()
        val = Crippen._GetAtomContribs(mol)[atom.GetIdx()][0]
        return val if val is not None else 0.0


@atom_features.register(name='crippen_molar_refractivity_contribution')
class CrippenMolarRefractivityContribution(AtomicFeature):
    def __call__(self, atom: Chem.Atom) -> float:
        mol = atom.GetOwningMol()
        val = Crippen._GetAtomContribs(mol)[atom.GetIdx()][1]
        return val if val is not None else 0.0


@atom_features.register(name='tpsa_contribution')
class TPSAContribution(AtomicFeature):
    def __call__(self, atom: Chem.Atom) -> float:
        mol = atom.GetOwningMol()
        val = rdMolDescriptors._CalcTPSAContribs(mol)[atom.GetIdx()]
        return val if val is not None else 0.0


@atom_features.register(name='labute_asa_contribution')
class LabuteASAContribution(AtomicFeature):
    def __call__(self, atom: Chem.Atom) -> float:
        mol = atom.GetOwningMol()
        val = rdMolDescriptors._CalcLabuteASAContribs(mol)[0][atom.GetIdx()]
        return val if val is not None else 0.0


@atom_features.register(name='gasteiger_charge')
class GasteigerCharge(AtomicFeature):
    def __call__(self, atom: Chem.Atom) -> float:
        mol = atom.GetOwningMol()
        rdPartialCharges.ComputeGasteigerCharges(mol)
        val = atom.GetDoubleProp('_GasteigerCharge')
        return val if val is not None else 0.0


@bond_features.register(name='bond_type')
class BondType(AtomicFeature):
    def __call__(self, bond: Chem.Bond) -> str:
        return bond.GetBondType().name


@bond_features.register(name='conjugated')
class Conjugated(AtomicFeature):
    def __call__(self, bond: Chem.Bond) -> bool:
        return bond.GetIsConjugated()


@bond_features.register(name='rotatable')
class Rotatable(AtomicFeature):
    def __call__(self, bond: Chem.Bond) -> bool:
        mol = bond.GetOwningMol()
        atom_indices = tuple(
            sorted([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]))
        return atom_indices in Lipinski._RotatableBonds(mol)


@bond_features.register(name='stereo')
class Stereo(AtomicFeature):
    def __call__(self, bond: Chem.Bond) -> str:
        return bond.GetStereo().name
