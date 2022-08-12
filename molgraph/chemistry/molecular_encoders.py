import numpy as np
from rdkit import Chem
import tensorflow as tf
from typing import Callable
from typing import Union
from typing import Optional
from typing import List
from typing import Any
from typing import Callable
from typing import Dict
from typing import Tuple
from typing import Sequence
import multiprocessing
from functools import partial
import logging
import collections

from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from molgraph.tensors.graph_tensor import GraphTensor
from molgraph.chemistry.atomic.featurizers import AtomFeaturizer
from molgraph.chemistry.atomic.featurizers import BondFeaturizer
from molgraph.chemistry.transform_ops import molecule_from_string


logger = logging.getLogger(__name__)


@dataclass
class BaseMolecularGraphEncoder(ABC):

    """This class takes as input an atom encoder and [optionally] bond encoder,
    as well as parameters specifying if positional encodings of the atoms should
    be computed; if self loops should be included (namely, if an atom should
    have a connection (bond) to itself); and what molecule_from_string_fn that should
    be used to convert SMILES/InChI to an RDKit molecule object.

    The resulting instance of the class can then be called (via its `__call__`,
    method or `encode` method) to convert a SMILES/InChI/RDKit mol to a
    molecular graph (namely, a `GraphTensor`). The `GraphTensor` is a custom
    tensor class (which inherits from `CompositeTensor`) and can be seamlessly
    used with TensorFlow/Keras.
    """


    atom_encoder: Union[AtomFeaturizer, Callable]

    @abstractmethod
    def call(
        self,
        molecule: Union[str, Chem.Mol],
        device: str = '/cpu:0',
        **kwargs,
    ) -> GraphTensor:
        pass

    def __call__(
        self,
        inputs: Sequence[Union[str, Chem.Mol]],
        processes: Optional[int] = None,
        device: str = '/cpu:0',
        **kwargs,
    ) -> GraphTensor:
        """Generates molecular graphs (GraphTensor) from a list of InChI/SMILES
        or RDKit molecule objects, with multiprocessing.
        """
        if isinstance(inputs, (list, tuple, set, np.ndarray)):

            # Convert a list of inputs to a list of `GraphTensor`s.
            # To lower the run-time, multiprocessing is used.
            with multiprocessing.Pool(processes) as pool:
                graph_tensor_list = pool.map(
                    func=partial(self.call, device=device, **kwargs),
                    iterable=inputs
                )

            graph_tensor_list = [
                gt for gt in graph_tensor_list if gt is not None]

            # The list of `GraphTensor`s is concatenated to generate a single
            # `GraphTensor` (disjoint [molecular] graph). The `separate` method
            # is called to make the nested structures of the `GraphTensor`
            # ragged. This will allow for batching of the `GraphTensor`.
            return tf.concat(graph_tensor_list, axis=0).separate()

        return self.call(inputs, device=device, **kwargs)


@dataclass
class MolecularGraphEncoder(BaseMolecularGraphEncoder):

    bond_encoder: Optional[Union[BondFeaturizer, Callable]] = None
    molecule_from_string_fn: Callable[[str], Chem.Mol] = field(
        default_factory=lambda: partial(molecule_from_string, catch_errors=True),
        repr=False
    )
    positional_encoding_dim: Optional[int] = 20
    self_loops: bool = False
    auxiliary_encoders: Optional[Dict[str, Callable]] = field(
        default=None, repr=False)

    def call(
        self,
        molecule: Union[str, Chem.Mol],
        device: str = '/cpu:0',
        index_dtype: str = 'int32'
    ) -> GraphTensor:

        """Generates a molecular graph (`GraphTensor`) from a given InChI
        string, SMILES string, or molecule object.
        """

        with tf.device(device):

            # Convert string (SMILES or InChI) to RDKit mol if necessary
            if not isinstance(molecule, (str, Chem.Mol)):
                raise ValueError(
                    "`molecule` needs to be a string or " +
                    "a RDKit molecule object (`Chem.Mol`), " +
                    "not {}".format(type(molecule))
                )

            if not isinstance(molecule, Chem.Mol):
                molecule = self.molecule_from_string_fn(molecule)

            if molecule is None:
                raise ValueError(
                    f"Could not convert input ({format}) to an RDKit mol")

            # Initialize data dictionary
            data = {}

            # Obtain destination and source node (atom) indices of edges (bonds)
            sparse_adjacency = _compute_adjacency(
                molecule, self.self_loops, sparse=True, dtype=index_dtype)
            data['edge_dst'], data['edge_src'] = sparse_adjacency

            # Obtain node (atom) features
            atoms = _get_atoms(molecule)
            data['node_feature'] = self.atom_encoder.encode_atoms(atoms)

            # Obtain edge (bond) features (if `bond_encoder` exist)
            if self.bond_encoder is not None:
                bonds = _get_bonds(molecule, *sparse_adjacency)
                data['edge_feature'] = self.bond_encoder.encode_bonds(bonds)

            # Obtain positional encoding of nodes (atoms)
            if self.positional_encoding_dim:
                data['positional_encoding'] = _compute_positional_encoding(
                    molecule=molecule,
                    dim=self.positional_encoding_dim,
                    dtype=getattr(self.atom_encoder, 'dtype', 'float32'))

            if self.auxiliary_encoders is not None:
                for field, encoder in self.auxiliary_encoders.items():
                    data[field] = encoder(molecule)

            return GraphTensor(data)


@dataclass
class MolecularGraphEncoder3D(BaseMolecularGraphEncoder):

    """Distance geometric molecular graph encoder."""

    conformer_generator: Optional[Callable] = None
    edge_radius: Optional[int] = None
    coloumb: bool = True

    def call(
        self,
        molecule: str,
        device: str = '/cpu:0',
        index_dtype: str = 'int32'
    ) -> GraphTensor:

        """Generates a 3D molecular graph (`GraphTensor`) from a given InChI,
        SMILES, SDF block, or RDKit molecule object.
        """

        with tf.device(device):

            if self.conformer_generator is not None:
                molecule = self.conformer_generator(molecule)
            else:
                molecule = molecule_from_string(molecule)

            if molecule is None:
                raise ValueError(
                    f"Could not convert input ({format}) to an RDKit mol")

            # Initialize data dictionary
            data = {}

            dg = _compute_distance_geometry(
                molecule, radius=self.edge_radius)

            atoms = _get_atoms(molecule)
            data['node_feature'] = self.atom_encoder.encode_atoms(atoms)
            data['edge_dst'] = np.array(dg['edge_dst'], dtype=index_dtype)
            data['edge_src'] = np.array(dg['edge_src'], dtype=index_dtype)

            if not self.coloumb:
                edge_feature = dg['edge_length']
            else:
                nuclear_charge = np.array([
                    atom.GetAtomicNum() for atom in atoms], dtype=np.float32)
                nuclear_charge_dst = np.take(nuclear_charge, dg['edge_dst'])
                nuclear_charge_src = np.take(nuclear_charge, dg['edge_src'])
                edge_feature = (
                    nuclear_charge_dst * nuclear_charge_src) / dg['edge_length']
                edge_feature = np.expand_dims(edge_feature, -1)

            data['edge_feature'] = np.array(edge_feature, dtype=np.float32)

            return GraphTensor(data)


def _get_atoms(molecule: Chem.Mol) -> List[Chem.Atom]:
    """Returns a list of atoms given an RDKit mol object.
    """
    return list(molecule.GetAtoms())

def _get_bonds(
    molecule: Chem.Mol,
    edge_dst: np.ndarray,
    edge_src: np.ndarray
) -> List[Chem.Bond]:
    """Returns a list of bonds given an RDKit mol object. The order of the
    bonds in the list corresponds to the sparse adjacency matrix which is
    also part of the resulting `GraphTensor`.
    """
    return [
        molecule.GetBondBetweenAtoms(int(i), int(j))
        for (i, j) in zip(edge_dst, edge_src)
    ]

def _compute_positional_encoding(
    molecule: Chem.Mol,
    dim: int = 20,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Computes Laplacian positional encoding from a RDKit molecule object.

    The laplacian positional encoding encodes the position of nodes (atoms) in
    the molecular graph. This could be seen as a replacement for the typical
    positional encoding of the transformer models for natural language
    processing tasks (e.g. BERT).
    """

    # If the molecule only has one atom, return a zero vector
    if molecule.GetNumAtoms() <= 1:
        return np.zeros((1, dim), dtype=dtype)

    # Compute the adjacency matrix of the molecule
    adjacency = _compute_adjacency(molecule, sparse=False)

    # Compute the laplacian matrix
    laplacian = _compute_laplacian(adjacency)

    # Compute the eigen vectors (sorted by its eigen values) of the laplcian.
    # This eigen vectors are the positional encoding.
    eigen_vectors = _compute_sorted_eigen_vectors(laplacian)

    # Remove the first element of the eigen vectors
    positional_encoding = eigen_vectors[:, 1: dim + 1]

    # Pad with zeros for molecules with number of atoms less than dim.
    positional_encoding = np.pad(
        positional_encoding, [
            (0, 0), (0, max(0, dim-positional_encoding.shape[1]))
        ]
    )
    return positional_encoding.astype(dtype)

def _compute_sorted_eigen_vectors(
    laplacian: np.ndarray,
    dtype: np.dtype = np.float32
) -> np.ndarray:
    """Computes eigen vectors of the laplacian matrix, sorted by the
    eigen values.
    """
    eigen_values, eigen_vectors = np.linalg.eig(laplacian)
    indices = eigen_values.argsort()
    eigen_values = eigen_values[indices]
    eigen_vectors = np.real(eigen_vectors[:, indices])
    return eigen_vectors

def _compute_adjacency(
    molecule: Chem.Mol,
    self_loops: bool = False,
    sparse: bool = False,
    dtype: np.dtype = np.int32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes an adjacency matrix or a sparse adjacency matrix from an
    RDKit molecule object.
    """

    adjacency = Chem.GetAdjacencyMatrix(molecule)

    if self_loops:
        adjacency += np.eye(adjacency.shape[0], dtype=adjacency.dtype)

    if not sparse:
        return adjacency.astype(dtype)

    bond_dst, bond_src = np.where(adjacency)

    return bond_dst.astype(dtype), bond_src.astype(dtype)

def _compute_laplacian(
    adjacency: np.ndarray,
    dtype: np.dtype = np.float32
) -> np.ndarray:
    """Computes the laplacian matrix an adjacency matrix
    """
    degree = np.sum(adjacency, axis=1)
    degree = np.sqrt(degree)
    degree = np.divide(1., degree, out=np.zeros_like(degree), where=degree!=0)
    degree = np.diag(degree)
    adjacency = degree.dot(adjacency).dot(degree)
    laplacian = np.eye(adjacency.shape[0]) - adjacency
    return laplacian.astype(dtype)

def _compute_distance_between_atoms(
    molecule: Chem.Mol,
    edge_dst: int,
    edge_src: int,
    unit: str
) -> float:
    '''Computes distance between two atoms in the molecule'''
    bond_length = Chem.rdMolTransforms.GetBondLength(
        molecule.GetConformer(), edge_dst, edge_src)
    if unit.lower() == 'bohr':
        return bond_length * 1.8897259885789
    return bond_length

def _compute_distance_geometry(
    molecule: Chem.Mol,
    radius: Optional[int] = None,
    unit: str = 'angstrom',
    atom: Union[Chem.Atom, None] = None,
    path: Union[List[int], None] = None,
    data: Union[Dict[str, List[Union[int, float]]], None] = None,
) -> Dict[str, List[Union[int, float]]]:

    '''Recursively navigates paths (in molecule) up to a certain `radius` to
    accumulate distance geometric information. If radius is None, the distance
    between every atom in the molecule will be computed.
    '''

    if path is None:
        data = {
            'edge_length': [], 'edge_dst': [], 'edge_src': [], 'edge_order': []}
        for atom in molecule.GetAtoms():
            path = _compute_distance_geometry(
                molecule, radius, unit, atom, [atom.GetIdx()], data)
        return data
    elif radius and len(path) > (radius + 1):
        return path[:-1]
    elif len(path) > 1:
        data['edge_dst'].append(path[0])
        data['edge_src'].append(path[-1])
        data['edge_length'].append(
            _compute_distance_between_atoms(molecule, path[0], path[-1], unit))
        data['edge_order'].append(-1 + len(path))

    for neighbor in atom.GetNeighbors():
        if neighbor.GetIdx() not in path:
            path.append(neighbor.GetIdx())
            path = _compute_distance_geometry(
                molecule, radius, unit, neighbor, path, data)
    return path[:-1]
