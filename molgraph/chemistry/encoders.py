import numpy as np

from rdkit import Chem

from abc import abstractmethod
from abc import ABC

import re
import hashlib

from typing import Union
from typing import List
from typing import Any

from molgraph.chemistry.features import Feature



class Featurizer:

    '''Atomic featurizer.

    Args:
        features (list[Feature]):
            List of atomic features.
        output_dtype (str, np.dtype):
            The output dtype.

    **Examples:**

    Atom featurizer:

    >>> atom_featurizer = molgraph.chemistry.Featurizer([
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

    >>> bond_featurizer = molgraph.chemistry.Featurizer([
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
        features: List[Feature],
        output_dtype: str = 'float32'
    ) -> None:
        self._feature_type = _validate_features(features)
        self.features = _wrap_features(features)
        self._nfeatures = len(self.features)
        self.output_dtype = output_dtype
        self._ndim = _get_ndim(self, self._feature_type)

    def __call__(
        self,
        inputs: Union[
            List[Chem.Atom],
            List[Union[Chem.Bond, None]],
            Chem.Atom,
            Union[Chem.Bond, None],
        ],
        *args,
        **kwargs,
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
        if inputs is not None and not isinstance(inputs, (Chem.Atom, Chem.Bond)): 
            inputs = list(inputs)

        if not isinstance(inputs, (list, tuple, set, np.ndarray)):
            return np.concatenate([
                feature(inputs) for feature in self.features
            ]).astype(self.output_dtype)

        if self._feature_type == 'atom':
            return self._encode_atoms(inputs, *args, **kwargs)
        return self._encode_bonds(inputs, *args, **kwargs)

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
        return f'Featurizer(features={self.features})'


class Tokenizer:

    '''Atomic tokenizer.

    Args:
        features (list[Feature]):
            List of atomic (atom or bond) features.

    **Example:**

    >>> atom_tokenizer = molgraph.chemistry.Tokenizer([
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
        self.features = _wrap_features(features, tokenize=True)

    def __call__(
        self,
        inputs: Union[
            List[Chem.Atom],
            List[Union[Chem.Bond, None]],
            Chem.Atom,
            Union[Chem.Bond, None],
        ],
        *args,
        **kwargs,
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
        
        if inputs is not None and not isinstance(inputs, (Chem.Atom, Chem.Bond)): 
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
        return f'Tokenizer(features={self.features})'


class Encoding(ABC):

    'Wraps around ``Feature`` to make it encodable.'

    @abstractmethod
    def __init__(self, feature: Feature) -> None:
        pass

    @abstractmethod
    def __call__(self, inputs: Union[Chem.Atom, Chem.Bond]) -> np.ndarray:
        pass

    def __getattr__(self, name):
        if hasattr(object.__getattribute__(self, '_wrapped_feature'), name):
            return getattr(self._wrapped_feature, name)
        return object.__getattribute__(self, name)

    def __repr__(self) -> str:
        return f"{self._wrapped_feature!r}"


class NominalEncoding(Encoding):

    def __init__(self, feature: Feature) -> None:
        self._wrapped_feature = feature
        keys = list(self._wrapped_feature.allowable_set)
        keys.sort(key=lambda x: x if x is not None else "")
        dim = len(keys) + self._wrapped_feature.oov_size

        for i in range(self._wrapped_feature.oov_size):
            keys.insert(0, f"[OOV:{i}]")

        values = np.eye(dim, dtype='float32')

        self.mapping = dict(zip(keys, values))

        if not self._wrapped_feature.oov_size:
            self.mapping["[OOV:0]"] = np.zeros(dim, 'float32')

        self._wrapped_feature.allowable_set = keys

    def __call__(self, inputs: Union[Chem.Atom, Chem.Bond]) -> np.ndarray:
        output = self._wrapped_feature(inputs)
        index = _compute_oov_index(output, self._wrapped_feature.oov_size)
        return self.mapping.get(
            self._wrapped_feature(inputs), self.mapping[f'[OOV:{index}]'])


class OrdinalEncoding(Encoding):

    def __init__(self, feature: Feature) -> None:
        self._wrapped_feature = _assert_ordered_collection(feature)
        keys = list(self._wrapped_feature.allowable_set)
        dim = len(keys)
        values = np.tril(np.ones(dim, dtype='float32'))
        self.mapping = dict(zip(keys, values))
        self.mapping["[OOV]"] = np.zeros(dim, 'float32')
        self._wrapped_feature.allowable_set = keys

    def __call__(self, inputs: Union[Chem.Atom, Chem.Bond]) -> np.ndarray:
        return self.mapping.get(
            self._wrapped_feature(inputs), self.mapping['[OOV]'])


class FloatEncoding(Encoding):

    def __init__(self, feature: Feature) -> None:
        self._wrapped_feature = feature

    def __call__(self, inputs: Union[Chem.Atom, Chem.Bond]) -> np.ndarray:
        return np.array([self._wrapped_feature(inputs)], dtype='float32')


class TokenEncoding(Encoding):

    def __init__(self, feature: Feature) -> None:
        self._wrapped_feature = feature
        self._wrapped_feature.__dict__.clear()
        self._feature_name = _camel_case(self._wrapped_feature.name, n=3) + ':'

    def __call__(self, inputs: Union[Chem.Atom, Chem.Bond]) -> str:
        inputs = self._wrapped_feature(inputs)
        if isinstance(inputs, (bool, float)):
            inputs = int(inputs)
        return self._feature_name + str(inputs)


def _compute_oov_index(inputs: Feature, oov_size: int) -> int:
    if oov_size == 0:
        return 0
    unicode_string = str(inputs).encode('utf-8')
    hash_integer = int(hashlib.md5(unicode_string).hexdigest(), 16)
    return hash_integer % oov_size

def _assert_ordered_collection(inputs: Feature) -> Feature:
    assert isinstance(inputs.allowable_set, (list, tuple, str)), (
        '`allowable_set` needs to be an ordered collection when `ordinal=True`')
    return inputs

def _camel_case(s: str, n: int = 3) -> str:
    s = re.sub('^(.)', lambda m: m.group(1).upper(), s)
    s = re.sub('[_]+(.)', lambda m: m.group(1).upper(), s)
    words = re.findall('[A-Z0-9][^A-Z0-9]*', s)
    return ''.join([w[:n] for w in words])

def _validate_features(features: List[Feature]):
    dummy_mol = Chem.MolFromSmiles('CC')
    dummy_atom = dummy_mol.GetAtomWithIdx(0)
    dummy_bond = dummy_mol.GetBondWithIdx(0)
    try:
        # Check if features are atom features
        for f in features:
            atom_feature_name = f.__class__.__name__
            _ = f(dummy_atom)
        else:
            atom_feature_name = ''
            feature_type = 'atom'
    except Exception as e_atom:
        try:
            # Check if features are bond features
            for f in features:
                bond_feature_name = f.__class__.__name__
                _ = f(dummy_bond)
            else:
                bond_feature_name = ''
                feature_type = 'bond'
        except Exception as e_bond:
            error_atom_message = f'{type(e_atom).__name__}: {str(e_atom)}'
            error_bond_message = f'{type(e_bond).__name__}: {str(e_bond)}'
            feature_type = None

    if feature_type is None:
        raise ValueError(
            'Could not compute atom or bond features:\n\t'
            f'- [atom] {atom_feature_name} raised {error_atom_message}\n\t'
            f'- [bond] {bond_feature_name} raised {error_bond_message}')

    return feature_type

def _wrap_features(
    features: List[Feature],
    tokenize: bool = False,
) -> List[Encoding]:
    wrapped_features = []
    for f in features:
        if tokenize:
            wrapped_features.append(TokenEncoding(f))
        elif not hasattr(f, 'allowable_set'):
            wrapped_features.append(FloatEncoding(f))
        elif getattr(f, 'ordinal', False):
            wrapped_features.append(OrdinalEncoding(f))
        else:
            wrapped_features.append(NominalEncoding(f))
    return wrapped_features

def _get_ndim(featurizer: Featurizer, feature_type: str):
    dummy_mol = Chem.MolFromSmiles('CC')
    if feature_type == 'atom':
        return len(featurizer(dummy_mol.GetAtomWithIdx(0)))
    return len(featurizer(dummy_mol.GetBondWithIdx(0)))

def _check_bond(bond: Any) -> None:
    if not isinstance(bond, Chem.Bond) and bond is not None:
        raise ValueError('bond needs to be either a `rdkit.Chem.Bond` or None')

def _check_atom(atom: Any) -> None:
    if not isinstance(atom, Chem.Atom):
        raise ValueError('atom needs to be a `rdkit.Chem.Atom`')
