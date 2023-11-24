import numpy as np

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

import math
import functools
import multiprocessing
import sys

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from collections import defaultdict

from typing import List
from typing import Optional
from typing import Union
from typing import Dict
from typing import Tuple
from typing import Any

from molgraph.chemistry.ops import molecule_from_string



def serialize(
    splitter: 'Splitter',
) -> Dict[str, Union[str, Dict[str, Any]]]:
    """Simplistic implementation to serialize a Splitter instance.
    """
    assert isinstance(splitter, Splitter), (
        "`splitter` is not a `Splitter` instance")
    serialized_splitter = {}
    serialized_splitter.update({
        'class_name': splitter.__class__.__name__})
    serialized_splitter.update({
        'config': splitter.__dict__})
    return serialized_splitter

def deserialize(
    serialized_splitter: Optional[
        Union['Splitter', Dict[str, Union[str, Dict[str, Any]]]]
    ]
) -> 'Splitter':
    """Simplistic implementation to deserialize a serialized Splitter instance.
    """
    if serialized_splitter is None:
        return None
    elif isinstance(serialized_splitter, dict):
        name = serialized_splitter['class_name']
        splitter = getattr(sys.modules[__name__], name)
        return splitter(**serialized_splitter['config'])
    else:
        return serialized_splitter


@dataclass
class Splitter(ABC):

    """Base class for splitters."""

    validation_size: Union[int, float] = 0.1
    test_size: Union[int, float] = 0.1
    seed: Optional[int] = None

    @abstractmethod
    def split(self, x, y=None, groups=None) -> Tuple[np.ndarray, ...]:
        pass

    def prepare_split(self, x: Any) -> Tuple[int, ...]:
        total_size = (
            len(x) if isinstance(x, (list, tuple, set)) else x.shape[0])
        if isinstance(self.test_size, float):
            validation_size = math.floor(
                (1 - self.test_size / (self.test_size + self.validation_size))
                * (self.validation_size + self.test_size)
                * total_size)
            test_size = math.floor(
                self.test_size / (self.test_size + self.validation_size)
                * (self.validation_size + self.test_size)
                * total_size)
        return total_size, validation_size, test_size


@dataclass
class RandomSplitter(Splitter):
    """Splits data randomly"""

    def split(self, x, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, ...]:
        total_size, validation_size, test_size = self.prepare_split(x)
        return _random_split(
            np.arange(total_size), validation_size, test_size, self.seed)


@dataclass
class StratifiedSplitter(Splitter):
    """Splits continuous-labelled data in a stratified way"""

    def split(self, x, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, ...]:
        total_size, validation_size, test_size = self.prepare_split(x)
        return _stratified_split_from_continuous_labels(
            y, total_size, validation_size, test_size, self.seed)


@dataclass
class ScaffoldSplitter(Splitter):
    """Splits data based on scaffold groups"""

    processes: Optional[int] = None

    def split(self, x, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, ...]:
        total_size, validation_size, test_size = self.prepare_split(x)
        groups = _compute_scaffold_groups(x, self.processes)
        train_size = total_size - validation_size - test_size
        return _sorted_group_split(
            groups, train_size, validation_size, test_size)


def _stratified_split_from_continuous_labels(
    y: np.ndarray,
    total_size: int,
    validation_size: int,
    test_size: int,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, ...]:

    """A stratified split on continuous labels. May also work well for sparse
    representation of categorical (discrete) labels (e.g., [0, 5, ..., 2, 2])

    Sorts the indices based on the labels (y); then samples it "chunk-wise",
    based on the gcd of `train_size`, `validation_size` and `test_size`.
    """

    def greatest_common_denominator(*nums) -> int:
        return functools.reduce(lambda x, y: math.gcd(x, y), nums)

    if seed is not None:
        np.random.seed(seed)

    sorted_indices = np.argsort(np.squeeze(y))

    train_size = total_size - validation_size - test_size

    gcd = greatest_common_denominator(
        train_size, validation_size, test_size)

    split_train_samples = train_size // gcd
    split_validation_samples = validation_size // gcd
    split_test_samples = test_size // gcd

    split_size = sum([
        split_train_samples,
        split_validation_samples,
        split_test_samples])

    train_index = []
    validation_index = []
    test_index = []
    for i in range(0, total_size, split_size):

        shuffled_indices = np.random.permutation(
            sorted_indices[i: i+split_size])

        train_index.append(
            shuffled_indices[:split_train_samples])

        validation_index.append(
            shuffled_indices[
                split_train_samples:
                split_train_samples + split_validation_samples])

        test_index.append(
            shuffled_indices[
                split_train_samples + split_validation_samples:])

    return (
        np.sort(np.concatenate(train_index)),
        np.sort(np.concatenate(validation_index)),
        np.sort(np.concatenate(test_index))
    )

def _random_split(
    indices: np.ndarray,
    validation_size: int,
    test_size: int,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray]:
    """A random split.

    First shuffles the indices and then splits it by based on validation and
    test sizes.
    """
    if seed is not None:
        np.random.seed(seed)
    random_indices = np.random.permutation(indices)
    test_index, validation_index, train_index = np.split(
        random_indices, [test_size, test_size + validation_size])
    return (
        np.sort(train_index),
        np.sort(validation_index),
        np.sort(test_index)
    )

def _sorted_group_split(
    groups: np.ndarray,
    train_size: int,
    validation_size: int,
    test_size: int,
) -> Tuple[np.ndarray]:

    """A deterministic group split.

    Sort groups by size (descending), then populates the training indices,
    followed by validation indices and finally test indices.
    """

    group_sets = {i: np.where(groups == i)[0] for i in range(max(groups))}
    group_sets = sorted(
        group_sets.items(),
        key=lambda x: (len(x[1]), x[1][-1]),
        reverse=True)
    group_sets = np.array(
        [group_set for (_, group_set) in group_sets], dtype=object)

    train_indices = []
    validation_indices = []
    test_indices = []
    for group_set in group_sets:
        if len(train_indices) + len(group_set) <= train_size:
            train_indices.extend(list(group_set))
        elif len(validation_indices) + len(group_set) <= validation_size:
            validation_indices.extend(list(group_set))
        else:
            test_indices.extend(list(group_set))

    return (
        np.sort(train_indices),
        np.sort(validation_indices),
        np.sort(test_indices)
    )

def _get_scaffold(molecule: str) -> str:

    """Computes the Murcko scaffold from a SMILES, InChI or SDF string.
    """

    if not isinstance(molecule, Chem.Mol):
        molecule = molecule_from_string(molecule, catch_errors=True)
    if molecule is None:
        return
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=molecule, includeChirality=False)
    except:
        scaffold = 'Undefined'
    return scaffold

def _compute_scaffold_groups(
    strings: List[str],
    processes: Optional[int] = None
) -> np.ndarray:

    """Computes Murcko scaffold groups from a list of SMILES, InChI or SDF
    strings. The groups can then be used to split datasets (x, y) by groups.

    Uses the multiprocessing module to compute scaffolds in parallel.
    """

    with multiprocessing.Pool(processes) as pool:
        scaffolds = pool.map(_get_scaffold, strings)

    scaffolds = [scaffold for scaffold in scaffolds if scaffold is not None]
    scaffold_groups = defaultdict(list)
    for index, scaffold in enumerate(scaffolds):
        scaffold_groups[scaffold].append(index)

    scaffold_sets = [
        scaffold_set for scaffold_set in scaffold_groups.values()]

    indices = np.array([i for j in scaffold_sets for i in j])
    repeats = [len(scaffold_set) for scaffold_set in scaffold_sets]
    scaffold_sets = np.arange(len(scaffold_sets))
    scaffold_sets = np.repeat(scaffold_sets, repeats)
    pairs = zip(indices, scaffold_sets)
    pairs = sorted(pairs, key=lambda x: x[0])
    return np.array([scaffold for (index, scaffold) in pairs])
