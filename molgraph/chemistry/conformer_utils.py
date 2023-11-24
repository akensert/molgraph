import numpy as np

from rdkit import Chem
from rdkit.Chem import rdDistGeom

import logging

from typing import Optional
from typing import List
from typing import Union


logger = logging.getLogger(__name__)


_embedding_method = {
    'ETDG': Chem.rdDistGeom.ETDG(),
    'ETKDG': Chem.rdDistGeom.ETKDG(),
    'ETKDGv2': Chem.rdDistGeom.ETKDGv2(),
    'ETKDGv3': Chem.rdDistGeom.ETKDGv3(),
    'srETKDGv3': Chem.rdDistGeom.srETKDGv3(),
    'KDG': Chem.rdDistGeom.KDG()
}

_available_ff_methods = [
    'MMFF', 'MMFF94', 'MMFF94s', 'UFF'
]


def embed_conformers(
    molecule: Chem.Mol,
    num_conformers: Optional[Union[int, str]] = 'auto',
    embedding_method: str = 'ETKDGv3',
    num_threads: int = 1,
    random_coords_threshold: int = 100,
) -> None:

    """Embeds one or more conformers from RDKit molecule object.
    """

    # Get copy, to leave original molecle untouched
    molecule = Chem.Mol(molecule)

    if num_conformers is None or num_conformers == 'auto':
        num_conformers = _get_num_conformers_from_molecule_size(molecule)

    params = _embedding_method.get(embedding_method, 'ETKDGv3')

    params.numThreads = num_threads

    if molecule.GetNumAtoms() > random_coords_threshold:
        # For large molecules, random coordinates might help
        params.useRandomCoords = True

    success = Chem.rdDistGeom.EmbedMultipleConfs(
        molecule, numConfs=num_conformers, params=params)

    if not len(success):
        # Not pretty, but it is desired to force a 3D structure (instead
        # of ignoring the molecule) so that the same molecules exist
        # between the 2d and 3d datasets.
        logger.warning(
            "Could not embed conformers, computing conformer from " +
            "2D coordinates instead."
        )
        Chem.rdDepictor.Compute2DCoords(molecule)

    return molecule

def force_field_minimize_conformers(
    molecule: Chem.Mol,
    force_field_method: Optional[str] = 'UFF',
    max_iter: Optional[Union[int, str]] = 'auto',
    return_energies: bool = False,
    num_threads: int = 1,
    **kwargs,
) -> Union[np.ndarray, None]:

    """Performs a force field minimization on embedded conformers of RDKit
    molecule object.
    """

    _assert_correct_force_field(force_field_method)

    # Get copy, to leave original molecle untouched
    molecule = Chem.Mol(molecule)

    if max_iter is None or max_iter == 'auto':
        max_iter = _get_max_iter_from_molecule_size(molecule)

    if force_field_method == 'MMFF':
        force_field_method = 'MMFF94'

    try:
        if force_field_method.startswith('MMFF'):
            # Merck Molecular Force Field (MMFF; specifically MMFF94 or MMFF94s)
            Chem.rdForceFieldHelpers.MMFFSanitizeMolecule(molecule)
            result = Chem.rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(
                molecule,
                maxIters=max_iter,
                numThreads=num_threads,
                mmffVariant=force_field_method,
                **kwargs)
        else:
            # Universal Force Field (UFF)
            result = Chem.rdForceFieldHelpers.UFFOptimizeMoleculeConfs(
                molecule, numThreads=num_threads, maxIters=max_iter, **kwargs)
    except RuntimeError:
        logger.warning(
            f"{force_field_method} raised a `RunTimeError`, proceeding " +
            f"without {force_field_method} minimization."
        )
        return molecule

    converged = [r[0] != 1 for r in result]

    if not any(converged):
        logger.warning(
            f"{force_field_method} minimization did not converge " +
            f"after {max_iter} iterations, for any of the " +
            f"{len(converged)} conformers."
        )

    if return_energies:
        return molecule, [r[1] for r in result]

    return molecule

def get_lowest_energy_conformer(molecule, force_field_method: str = 'UFF'):

    _assert_has_conformers(molecule)

    conformer_ids = [
        conformer.GetId() for conformer in molecule.GetConformers()
    ]
    energy_lowest = float('inf')
    for conformer_id in conformer_ids:
        energy = compute_force_field_energy(
            molecule, conformer_id, force_field_method)
        if energy < energy_lowest:
            energy_lowest = energy
            conformer_id_keep = conformer_id

    new_molecule = Chem.Mol(molecule)
    new_molecule.RemoveAllConformers()
    conformer = molecule.GetConformer(conformer_id_keep)
    new_molecule.AddConformer(conformer, assignId=True)
    return new_molecule

def prune_conformers(
    molecule: Chem.Mol,
    num_candidates_keep: Optional[int] = None,
    force_field_method: str = 'UFF',
    rmsd_threshold: float = 0.5,
) -> Chem.Mol:

    if num_candidates_keep is None:
        num_candidates_keep = molecule.GetNumConformers()

    rmsd = align_conformers(molecule)

    conformer_ids = [
        conformer.GetId() for conformer in molecule.GetConformers()
    ]
    energies = []
    for conformer_id in conformer_ids:
        ff_energy = compute_force_field_energy(
            molecule, conformer_id, force_field_method=force_field_method)
        energies.append(ff_energy)

    keep_conformer_ids = filter_conformers(
        molecule,
        num_candidates_keep=num_candidates_keep,
        energies=energies,
        rmsd=rmsd,
        rmsd_threshold=rmsd_threshold)

    new_molecule = Chem.Mol(molecule)
    new_molecule.RemoveAllConformers()
    for conformer_id in keep_conformer_ids:
        conformer = molecule.GetConformer(conformer_id)
        new_molecule.AddConformer(conformer, assignId=True)

    return new_molecule

def compute_force_field_energy(
    molecule: Chem.Mol,
    conformer_id: int,
    force_field_method: str = 'UFF'
) -> float:

    if force_field_method == 'MMFF':
        force_field_method = 'MMFF94'

    if force_field_method.startswith('MMFF'):
        mmff_properties = Chem.rdForceFieldHelpers.MMFFGetMoleculeProperties(
            mol=molecule, mmffVariant=force_field_method)
        force_field = Chem.rdForceFieldHelpers.MMFFGetMoleculeForceField(
            molecule, mmff_properties, confId=conformer_id)
    else:
        force_field = Chem.rdForceFieldHelpers.UFFGetMoleculeForceField(
            molecule, confId=conformer_id)

    return force_field.CalcEnergy()

def filter_conformers(
    molecule: Chem.Mol,
    num_candidates_keep: int,
    energies: np.ndarray,
    rmsd: np.ndarray,
    rmsd_threshold: float = 0.5,
) -> List[int]:
    """Filters conformers based on their energies and root mean squared
    deviation from each other. The purpose is to prune off conformers that
    are similar to each other (given by the `threshold`) while keep the
    low energy conformers.
    """
    _assert_has_conformers(molecule)

    conformer_indices = [
        conformer.GetId() for conformer in molecule.GetConformers()]

    keep_conformer_ids = []
    keep_ids = []
    for i, j in enumerate(np.argsort(energies)):
        if i == 0:
            keep_conformer_ids.append(conformer_indices[j])
            keep_ids.append(j)
            continue
        elif len(keep_ids) == num_candidates_keep:
            break

        rmsd_keep = rmsd[j, keep_ids]

        if np.all(rmsd_keep >= rmsd_threshold):
            keep_conformer_ids.append(conformer_indices[j])
            keep_ids.append(j)

    return keep_conformer_ids

def unpack_conformers(molecule: Chem.Mol) -> List[Chem.Mol]:
    """Inputs an RDKit molecule object and extracts its conformers. Each
    conformer is packed into a separate RDKit molecule object.
    """
    _assert_has_conformers(molecule)
    new_molecules = []
    for conformer in molecule.GetConformers():
        new_molecule = Chem.Mol(molecule)
        new_molecule.RemoveAllConformers()
        new_molecule.AddConformer(conformer, assignId=True)
        new_molecules.append(new_molecule)
    return new_molecules

def sort_conformers(
    molecule: Chem.Mol,
    energies: np.ndarray,
    ascending: bool = True
) -> Chem.Mol:
    """Sorts the conformers of molecule by their energies. Returns a new
    RDKit molecule object containing the same conformers of the inputted
    RDKit molecule object, but in sorted order.
    """
    _assert_has_conformers(molecule)
    conformer_ids = [
        conformer.GetId() for conformer in molecule.GetConformers()]
    new_molecule = Chem.Mol(molecule)
    sorted_energies = []
    new_molecule.RemoveAllConformers()
    for i in np.argsort(energies):
        sorted_energies.append(energies[i])
        conformer = molecule.GetConformer(conformer_ids[i])
        new_molecule.AddConformer(conformer, assignId=True)
    return new_molecule, sorted_energies

def align_conformers(molecule: Chem.Mol, dtype: np.dtype = np.float32) -> np.ndarray:
    """Computes the root mean squared deviation between the conformers
    of `molecule`. `molecule` needs to be a RDKit molecule object with
    at least one conformer.
    """

    _assert_has_conformers(molecule)

    num_conformers = molecule.GetNumConformers()
    rmsd = np.zeros((num_conformers, num_conformers), dtype=dtype)
    for i, ref_conformer in enumerate(molecule.GetConformers()):
        for j, prb_conformer in enumerate(molecule.GetConformers()):
            if i >= j:
                continue
            rmsd[[i, j], [j, i]] = Chem.rdMolAlign.GetBestRMS(
                prbMol=molecule,
                refMol=molecule,
                prbId=prb_conformer.GetId(),
                refId=ref_conformer.GetId())
    return rmsd


def _get_num_conformers_from_molecule_size(
    molecule: Chem.Mol,
    max_num_conformers: int = 10,
    min_num_conformers: int = 2,
    decr_num_conformers: int = 0.04,
) -> int:
    num_atoms = molecule.GetNumAtoms()
    return max(
        min_num_conformers,
        int(
            max_num_conformers -
            min(max_num_conformers-1, num_atoms * decr_num_conformers)
        )
    )

def _get_max_iter_from_molecule_size(
    molecule: Chem.Mol,
    min_iter: int = 20,
    max_iter: int = 2000,
    incr_iter: int = 10,
) -> int:
    return min(max_iter, int(min_iter + incr_iter * molecule.GetNumAtoms()))

def _assert_correct_force_field(force_field):
    if force_field not in _available_ff_methods:
        raise ValueError(
            f"`force_field_method` has to be either of: {_available_ff_methods}"
        )

def _assert_has_conformers(molecule):
    if not molecule.GetNumConformers():
        raise ValueError("`molecule` has no conformers.")
