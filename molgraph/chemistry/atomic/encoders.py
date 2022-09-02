import numpy as np
from rdkit import Chem
from abc import ABC
from abc import abstractmethod
from typing import Union
import re
import logging
import hashlib

from molgraph.chemistry.atomic.features import AtomicFeature


class AtomicEncoder(ABC):

    'Wraps around ``AtomicFeature`` to make it encodable.'

    @abstractmethod
    def __init__(self, feature: AtomicFeature) -> None:
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


class NominalEncoder(AtomicEncoder):

    def __init__(self, feature: AtomicFeature) -> None:
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


class OrdinalEncoder(AtomicEncoder):

    def __init__(self, feature: AtomicFeature) -> None:
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


class FloatEncoder(AtomicEncoder):

    def __init__(self, feature: AtomicFeature) -> None:
        self._wrapped_feature = feature

    def __call__(self, inputs: Union[Chem.Atom, Chem.Bond]) -> np.ndarray:
        return np.array([self._wrapped_feature(inputs)], dtype='float32')


class TokenEncoder(AtomicEncoder):

    def __init__(self, feature: AtomicFeature) -> None:
        self._wrapped_feature = feature
        self._wrapped_feature.__dict__.clear()
        self._feature_name = _camel_case(self._wrapped_feature.name, n=3) + ':'

    def __call__(self, inputs: Union[Chem.Atom, Chem.Bond]) -> str:
        inputs = self._wrapped_feature(inputs)
        if isinstance(inputs, (bool, float)):
            inputs = int(inputs)
        return self._feature_name + str(inputs)


def _compute_oov_index(inputs: AtomicFeature, oov_size: int) -> int:
    if oov_size == 0:
        return 0
    unicode_string = str(inputs).encode('utf-8')
    hash_integer = int(hashlib.md5(unicode_string).hexdigest(), 16)
    return hash_integer % oov_size

def _assert_ordered_collection(inputs: AtomicFeature) -> AtomicFeature:
    assert isinstance(inputs.allowable_set, (list, tuple, str)), (
        '`allowable_set` needs to be an ordered collection when `ordinal=True`')
    return inputs

def _camel_case(s: str, n: int = 3) -> str:
    s = re.sub('^(.)', lambda m: m.group(1).upper(), s)
    s = re.sub('[_]+(.)', lambda m: m.group(1).upper(), s)
    words = re.findall('[A-Z0-9][^A-Z0-9]*', s)
    return ''.join([w[:n] for w in words])
