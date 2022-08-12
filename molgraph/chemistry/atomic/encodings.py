import numpy as np
from dataclasses import dataclass
from dataclasses import field
from typing import Union
from typing import List
from typing import Dict
from rdkit import Chem
from abc import abstractmethod
from abc import ABC
import re
import logging
import hashlib

from molgraph.chemistry.atomic import features



class AtomicEncoding(ABC):

    '''
    Wraps around an `AtomicFeature`, allowing the `AtomicFeature` to be
    appropriately encoded in the molecular graph.
    '''

    @abstractmethod
    def __init__(self, feature: features.AtomicFeature) -> None:
        pass

    @abstractmethod
    def __call__(self, x: Union[Chem.Atom, Chem.Bond]) -> np.ndarray:
        pass

    def __repr__(self):
        return f"{self.feature!r}"


class NominalEncoding(AtomicEncoding):

    def __init__(self, feature: features.AtomicFeature) -> None:
        _check_attributes(feature)
        self.feature = feature
        keys = list(self.feature.allowable_set)
        keys.sort(key=lambda x: x if x is not None else "")
        dim = len(keys) + self.feature.oov_size

        for i in range(self.feature.oov_size):
            keys.insert(0, f"[OOV:{i}]")

        values = np.eye(dim, dtype='float32')

        self.mapping = dict(zip(keys, values))

        if "[OOV:0]" not in self.mapping:
            # if oov_size == 0
            self.mapping["[OOV:0]"] = np.zeros(dim, 'float32')

    def __call__(self, x: Union[Chem.Atom, Chem.Bond]) -> np.ndarray:
        output = self.feature(x)
        id = _compute_oov_id(output, self.feature.oov_size)
        return self.mapping.get(self.feature(x), self.mapping[f'[OOV:{id}]'])


class OrdinalEncoding(AtomicEncoding):

    def __init__(self, feature: features.AtomicFeature) -> None:
        _check_attributes(feature)
        self.feature = feature
        keys = self.feature.allowable_set
        dim = len(keys)
        values = np.tril(np.ones(dim, dtype='float32'))
        self.mapping = dict(zip(keys, values))
        self.mapping["[OOV]"] = np.zeros(dim, 'float32')

    def __call__(self, x: Union[Chem.Atom, Chem.Bond]) -> np.ndarray:
        return self.mapping.get(self.feature(x), self.mapping['[OOV]'])


class FloatEncoding(AtomicEncoding):

    def __init__(self, feature: features.AtomicFeature) -> None:
        self.feature = feature
        self.feature.allowable_set = None
        self.feature.ordinal = None
        self.oov_size = None

    def __call__(self, x: Union[Chem.Atom, Chem.Bond]) -> np.ndarray:
        return np.array([self.feature(x)], dtype='float32')


class TokenEncoding(AtomicEncoding):

    def __init__(self, feature: features.AtomicFeature) -> None:
        self.feature = feature
        self.feature.allowable_set = None
        self.feature.ordinal = None
        self.oov_size = None
        self._name = _camel_case(self.feature.name, n=3) + ':'

    def __call__(self, x: Union[Chem.Atom, Chem.Bond]) -> str:
        x = self.feature(x)
        if isinstance(x, (bool, float)):
            x = int(x)
        return self._name + str(x)


def _compute_oov_id(x, oov_size):
    if oov_size == 0:
        return 0
    unicode_string = str(x).encode('utf-8')
    hash_integer = int(hashlib.md5(unicode_string).hexdigest(), 16)
    return hash_integer % oov_size

def _check_attributes(feature):
    if isinstance(feature.allowable_set, set) and feature.ordinal:
        raise ValueError(
            f"({feature.name}) `allowable_set` needs to be an ordered " +
            "sequence (e.g. a `list`) when `ordinal` is set to True."
        )

def _camel_case(s: str, n: int = 3) -> str:
    s = re.sub('^(.)', lambda m: m.group(1).upper(), s)
    s = re.sub('[_]+(.)', lambda m: m.group(1).upper(), s)
    words = re.findall('[A-Z0-9][^A-Z0-9]*', s)
    return ''.join([w[:n] for w in words])
