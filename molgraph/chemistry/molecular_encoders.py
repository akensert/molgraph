import numpy as np

from rdkit import Chem

import tensorflow as tf

from typing import Callable
from typing import Union
from typing import Optional
from typing import List
from typing import Callable
from typing import Dict
from typing import Tuple
from typing import Sequence

import multiprocessing

from functools import partial

import logging

from dataclasses import dataclass
from dataclasses import field

from abc import ABC
from abc import abstractmethod

from molgraph.tensors.graph_tensor import GraphTensor
from molgraph.chemistry.encoders import Featurizer
from molgraph.chemistry.encoders import Tokenizer
from molgraph.chemistry.ops import molecule_from_string


logger = logging.getLogger(__name__)


@dataclass
class BaseMolecularGraphEncoder(ABC):

    'Base class for ``MolecularGraphEncoder`` and ``MolecularGraphEncoder3D``.'

    atom_encoder: Union[Featurizer, Tokenizer, Callable]

    @abstractmethod
    def call(self, molecule, **kwargs) -> GraphTensor:
        pass

    def __call__(
        self,
        inputs: Sequence[Union[str, Chem.Mol]],
        *,
        processes: Optional[int] = None,
        device: str = '/cpu:0',
        **kwargs,
    ) -> GraphTensor:
        '''Generates a molecular graph, namely ``GraphTensor``, from a molecule.
        Depending on ``molecule_from_string_fn`` the molecule(s) could be
        represented as SMILES, InChI or SDF files, etc.

        Args:
            inputs (str, list[str], Chem.Mol, list[Chem.Mol]):
                Molecules to be encoded as molecular graphs. Can either be
                a single molecule, or a list of molecules. In either case,
                a single ``GraphTensor`` will be obtained. The ``GraphTensor``
                has the flexibility to encode either a single molecule or
                multiple molecules.
            processes (int, None):
                The number of worker processes to use.
                If None ``os.cpu_count()`` is used. Default to None.
            device (str):
                Which device to use for generating the molecular graph.
                Default to '/cpu:0'.
            **kwargs:
                Any extra (keyword) arguments that may be used by the derived
                class. E.g., ``MolecularGraphEncoder`` passes ``index_dtype``
                to specify the dtype of node (atom) indices.

        Returns:
            GraphTensor: A single ``GraphTensor`` representing the
            molecule(s) inputted.
        '''
        with tf.device(device):

            if isinstance(inputs, (list, tuple, set, np.ndarray)):
                # Convert a list of inputs to a list of `GraphTensor`s.
                # To lower the run-time, multiprocessing is used.
                with multiprocessing.Pool(processes) as pool:
                    graph_tensor_list = pool.map(
                        func=partial(self.call, **kwargs),
                        iterable=inputs)

                graph_tensor_list = [
                    gt for gt in graph_tensor_list if gt is not None]

                return tf.concat(graph_tensor_list, axis=0)

            return self.call(inputs, **kwargs)


@dataclass
class MolecularGraphEncoder(BaseMolecularGraphEncoder):

    '''Molecular graph encoder, encoding molecular graphs as ``GraphTensor``.

    Args:
        atom_encoder (Featurizer, Tokenizer):
            The atom encoder to use.
        bond_encoder (Featurizer, Tokenizer, None):
            The bond encoder to use. Default to None.
        molecule_from_string_fn (callable):
            A function that produces an RDKit molecule object from some input,
            e.g. SMILES, InChI or SDFs. Default to
            ``chemistry.molecule_from_string``.
        positional_encoding_dim (int, None):
            The dimension of the positional encoding. If None, positional
            encoding will not be used. Default to 16.
        self_loops (bool):
            Whether self loops should be added to the molecular graph. Default
            to False.
        auxiliary_encoders: (dict[str, callable], None):
            Additional encoders to use to compute additional fields for the
            molecular graph. The outer dimension of the outputs of these
            encoders should match that of the outer dimension of the output
            of either the atom encoder or bond encoder. Default to None

    **Examples:**

    Generate a molecular graph with featurizers:

    >>> # Define atom featurizer (to produce numerical encoding of atoms)
    >>> atom_featurizer = molgraph.chemistry.Featurizer([
    ...     molgraph.chemistry.features.Symbol(),
    ...     molgraph.chemistry.features.Hybridization()
    ...     # ...
    ... ])
    >>> # Define bond featurizer (to produce numerical encoding of bonds)
    >>> bond_featurizer = molgraph.chemistry.Featurizer([
    ...     molgraph.chemistry.features.BondType(),
    ...     # ...
    ... ])
    >>> # Define molecular graph encoder
    >>> encoder = molgraph.chemistry.MolecularGraphEncoder(
    ...     atom_encoder=atom_featurizer,
    ...     bond_encoder=bond_featurizer,
    ...     positional_encoding_dim=10,
    ...     self_loops=False
    ... )
    >>> # Encode two molecules as a GraphTensor
    >>> graph_tensor = encoder(['CCC', 'CCO'])
    >>> graph_tensor
    GraphTensor(
      sizes=<tf.Tensor: shape=(2,), dtype=int32>,
      node_feature=<tf.Tensor: shape=(6, 119), dtype=float32>,
      edge_src=<tf.Tensor: shape=(8,), dtype=int32>,
      edge_dst=<tf.Tensor: shape=(8,), dtype=int32>,
      edge_feature=<tf.Tensor: shape=(8, 4), dtype=float32>,
      node_position=<tf.Tensor: shape=(6, 10), dtype=float32>)

    Generate a molecular graph with tokenizers:

    >>> # Define bond featurizer (to produce numerical encoding of atoms)
    >>> atom_tokenizer = molgraph.chemistry.Tokenizer([
    ...     molgraph.chemistry.features.Symbol(),
    ...     molgraph.chemistry.features.Hybridization()
    ...     # ...
    ... ])
    >>> # Define bond featurizer (to produce numerical encoding of bonds)
    >>> bond_tokenizer = molgraph.chemistry.Tokenizer([
    ...     molgraph.chemistry.features.BondType(),
    ...     # ...
    ... ])
    >>> # Define molecular graph encoder
    >>> encoder = molgraph.chemistry.MolecularGraphEncoder(
    ...     atom_encoder=atom_tokenizer,
    ...     bond_encoder=bond_tokenizer,
    ...     positional_encoding_dim=10,
    ...     self_loops=False
    ... )
    >>> # Encode two molecules as a GraphTensor
    >>> graph_tensor = encoder(['CCC', 'CCO'])
    >>> graph_tensor
    GraphTensor(
      sizes=<tf.Tensor: shape=(2,), dtype=int32>,
      node_feature=<tf.Tensor: shape=(6,), dtype=string>,
      edge_src=<tf.Tensor: shape=(8,), dtype=int32>,
      edge_dst=<tf.Tensor: shape=(8,), dtype=int32>,
      edge_feature=<tf.Tensor: shape=(8,), dtype=string>,
      node_position=<tf.Tensor: shape=(6, 10), dtype=float32>)

    Obtain numerical encodings of atoms (``node_feature``) and bonds
    (``bond_feature``) with the EmbeddingLookup layer. This is only necessary
    when tokenizers are used to compute ``node_feature`` and ``edge_feature``:

    >>> # Define bond featurizer (to produce numerical encoding of atoms)
    >>> atom_tokenizer = molgraph.chemistry.Tokenizer([
    ...    molgraph.chemistry.features.Symbol(),
    ...    molgraph.chemistry.features.Hybridization()
    ... ])
    >>> # Define bond featurizer (to produce numerical encoding of bonds)
    >>> bond_tokenizer = molgraph.chemistry.Tokenizer([
    ...    molgraph.chemistry.features.BondType(),
    ... ])
    >>> # Define molecular graph encoder
    >>> encoder = molgraph.chemistry.MolecularGraphEncoder(
    ...    atom_encoder=atom_tokenizer,
    ...    bond_encoder=bond_tokenizer,
    ...    positional_encoding_dim=10,
    ...    self_loops=False
    ... )
    >>> # Encode two molecules as a GraphTensor
    >>> graph_tensor = encoder(['CCC', 'CCO'])
    >>> # Define embedding layers
    >>> node_embedding = molgraph.layers.EmbeddingLookup(
    ...    feature='node_feature', output_dim=16)
    >>> edge_embedding = molgraph.layers.EmbeddingLookup(
    ...    feature='edge_feature', output_dim=8)
    >>> # Adapt embedding layers
    >>> node_embedding.adapt(graph_tensor)
    >>> edge_embedding.adapt(graph_tensor)
    >>> # Build model
    >>> model = tf.keras.Sequential([
    ...    node_embedding,
    ...    edge_embedding,
    ... ])
    >>> # Pass GraphTensor to model
    >>> graph_tensor = model(graph_tensor)
    >>> graph_tensor
    GraphTensor(
      sizes=<tf.Tensor: shape=(2,), dtype=int32>,
      node_feature=<tf.Tensor: shape=(6, 16), dtype=float32>,
      edge_src=<tf.Tensor: shape=(8,), dtype=int32>,
      edge_dst=<tf.Tensor: shape=(8,), dtype=int32>,
      edge_feature=<tf.Tensor: shape=(8, 8), dtype=float32>,
      node_position=<tf.Tensor: shape=(6, 10), dtype=float32>)
    '''

    bond_encoder: Optional[Union[
        Featurizer, Tokenizer, Callable]] = None
    molecule_from_string_fn: Callable[[str], Chem.Mol] = field(
        default_factory=lambda: partial(molecule_from_string, catch_errors=True),
        repr=False
    )
    positional_encoding_dim: Optional[int] = 16
    self_loops: bool = False
    auxiliary_encoders: Optional[Dict[str, Callable]] = field(
        default=None, repr=False)

    def call(
        self,
        molecule: Union[str, Chem.Mol],
        *,
        index_dtype: str = 'int32'
    ) -> GraphTensor:

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
        data['edge_src'], data['edge_dst'] = sparse_adjacency

        # Obtain node (atom) features
        atoms = _get_atoms(molecule)
        data['node_feature'] = self.atom_encoder(atoms)
        # Obtain edge (bond) features (if `bond_encoder` exist)
        if self.bond_encoder is not None:
            bonds = _get_bonds(molecule, *sparse_adjacency)
            data['edge_feature'] = self.bond_encoder(
                bonds, self_loops=self.self_loops)

        # Obtain positional encoding of nodes (atoms)
        if self.positional_encoding_dim:
            data['node_position'] = _compute_positional_encoding(
                molecule=molecule,
                dim=self.positional_encoding_dim,
                dtype=getattr(self.atom_encoder, 'dtype', 'float32'))

        if self.auxiliary_encoders is not None:
            for field, encoder in self.auxiliary_encoders.items():
                data[field] = encoder(molecule)

        return GraphTensor(**data)


@dataclass
class MolecularGraphEncoder3D(BaseMolecularGraphEncoder):

    '''Distance geometric molecular graph encoder, encoding molecular graphs
    as ``GraphTensor``.

    Args:
        atom_encoder (Featurizer, Tokenizer):
            The atom encoder to use.
        molecule_from_string_fn (callable):
            A function that produces an RDKit molecule object from some input,
            e.g. SMILES, InChI or SDFs. Default to
            ``chemistry.molecule_from_string``.
        conformer_generator (ConformerGenerator, callable, None):
            A conformer generator which produces a conformer of a given
            molecule, if a conformer does not exist. Default to None.
        edge_radius (int, None):
            The order of neighbors to consider for the distance geometry.
            If None, all atom pairs will be considered. Default to None.
        coulomb (bool):
            Whether coulomb values should be computed from the distances, and
            the associated atomic charges of the atom pairs. Default to True.

    **Examples:**

    >>> # Define bond featurizer (to produce numerical encoding of atoms)
    >>> atom_featurizer = molgraph.chemistry.Featurizer([
    ...     molgraph.chemistry.features.Symbol(),
    ...     molgraph.chemistry.features.Hybridization()
    ...     # ...
    ... ])
    >>> # Define conformer generator.
    >>> conformer_generator = molgraph.chemistry.ConformerGenerator()
    >>> # Define molecular graph encoder
    >>> encoder = molgraph.chemistry.MolecularGraphEncoder3D(
    ...     atom_encoder=atom_featurizer,
    ...     conformer_generator=conformer_generator,
    ...     edge_radius=None,
    ...     coulomb=False,
    ... )
    >>> # Encode two molecules as a GraphTensor
    >>> graph_tensor = encoder(['CCC', 'CCO'])
    >>> # The main difference between the 2d and 3d encoder is
    >>> # the edge_feature field. Here, in contains coulomb values,
    >>> # which mimics electrostatic interactions between nuclei
    >>> graph_tensor.edge_feature
    <tf.Tensor: shape=(12, 1), dtype=float32, numpy=
    array([[1.525636 ],
           [2.5192354],
           [1.525636 ],
           [1.5256361],
           [1.5256361],
           [2.5192354],
           [1.5208266],
           [2.3878794],
           [1.5208266],
           [1.3999726],
           [1.3999726],
           [2.3878794]], dtype=float32)>
    '''

    conformer_generator: Optional[Callable] = None
    molecule_from_string_fn: Callable[[str], Chem.Mol] = field(
        default_factory=lambda: partial(molecule_from_string, catch_errors=True),
        repr=False
    )
    edge_radius: Optional[int] = None
    coulomb: bool = True

    def call(
        self,
        molecule: str,
        *,
        index_dtype: str = 'int32'
    ) -> GraphTensor:

        if self.conformer_generator is not None:
            molecule = self.conformer_generator(molecule)
        else:
            molecule = self.molecule_from_string_fn(molecule)

        if molecule is None:
            raise ValueError(
                f"Could not convert input ({format}) to an RDKit mol")

        # Initialize data dictionary
        data = {}

        dg = _compute_distance_geometry(
            molecule, radius=self.edge_radius)

        atoms = _get_atoms(molecule)
        data['node_feature'] = self.atom_encoder(atoms)
        data['edge_src'] = np.array(dg['edge_src'], dtype=index_dtype)
        data['edge_dst'] = np.array(dg['edge_dst'], dtype=index_dtype)
        
        if not self.coulomb:
            edge_feature = np.expand_dims(dg['edge_length'], -1)
        else:
            nuclear_charge = np.array([
                atom.GetAtomicNum() for atom in atoms], dtype=np.float32)
            nuclear_charge_src = np.take(nuclear_charge, dg['edge_src'])
            nuclear_charge_dst = np.take(nuclear_charge, dg['edge_dst'])
            edge_feature = (
                nuclear_charge_dst * nuclear_charge_src) / dg['edge_length']
            edge_feature = np.expand_dims(edge_feature, -1)

        data['edge_feature'] = np.array(edge_feature, dtype=np.float32)

        return GraphTensor(**data)


def _get_atoms(molecule: Chem.Mol) -> List[Chem.Atom]:
    'Returns a list of atoms given an RDKit mol object.'
    return list(molecule.GetAtoms())

def _get_bonds(
    molecule: Chem.Mol,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
) -> List[Chem.Bond]:
    '''Returns a list of bonds given an RDKit mol object. The order of the
    bonds in the list corresponds to the sparse adjacency matrix which is
    also part of the resulting `GraphTensor`.
    '''
    return [
        molecule.GetBondBetweenAtoms(int(i), int(j))
        for (i, j) in zip(edge_src, edge_dst)
    ]

def _compute_positional_encoding(
    molecule: Chem.Mol,
    dim: int = 20,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    '''Computes Laplacian positional encoding from a RDKit molecule object.

    The laplacian positional encoding encodes the position of nodes (atoms) in
    the molecular graph. This could be seen as a replacement for the
    positional encoding of the transformer models for natural language
    processing tasks (e.g. BERT).
    '''

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
    '''Computes eigen vectors of the laplacian matrix, sorted by the
    eigen values.
    '''
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
    'Computes (sparse) adjacency matrix from an RDKit molecule object.'

    adjacency = Chem.GetAdjacencyMatrix(molecule)

    if self_loops:
        adjacency += np.eye(adjacency.shape[0], dtype=adjacency.dtype)

    if not sparse:
        return adjacency.astype(dtype)

    edge_src, edge_dst = np.where(adjacency)

    return edge_src.astype(dtype), edge_dst.astype(dtype)

def _compute_laplacian(
    adjacency: np.ndarray,
    dtype: np.dtype = np.float32
) -> np.ndarray:
    'Computes the laplacian matrix an adjacency matrix.'
    degree = np.sum(adjacency, axis=1)
    degree = np.sqrt(degree)
    degree = np.divide(1., degree, out=np.zeros_like(degree), where=degree!=0)
    degree = np.diag(degree)
    adjacency = degree.dot(adjacency).dot(degree)
    laplacian = np.eye(adjacency.shape[0]) - adjacency
    return laplacian.astype(dtype)

def _compute_distance_between_atoms(
    molecule: Chem.Mol,
    edge_src: int,
    edge_dst: int,
    unit: str
) -> float:
    'Computes distance between two atoms in the molecule.'
    edge_length = Chem.rdMolTransforms.GetBondLength(
        molecule.GetConformer(), edge_src, edge_dst)
    if unit.lower() == 'bohr':
        return edge_length * 1.8897259885789
    return edge_length

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
            'edge_length': [], 'edge_src': [], 'edge_dst': [], 'edge_order': []
        }
        for atom in molecule.GetAtoms():
            path = _compute_distance_geometry(
                molecule, radius, unit, atom, [atom.GetIdx()], data
            )
        return data
    elif radius and len(path) > (radius + 1):
        return path[:-1]
    elif len(path) > 1:
        data['edge_src'].append(path[0])
        data['edge_dst'].append(path[-1])
        data['edge_length'].append(
            _compute_distance_between_atoms(molecule, path[0], path[-1], unit)
        )
        data['edge_order'].append(-1 + len(path))

    for neighbor in atom.GetNeighbors():
        if neighbor.GetIdx() not in path:
            path.append(neighbor.GetIdx())
            path = _compute_distance_geometry(
                molecule, radius, unit, neighbor, path, data)
    return path[:-1]
