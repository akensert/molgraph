import numpy as np
import os

from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import SimilarityMaps

from PIL.PngImagePlugin import PngImageFile
from PIL import Image
from io import BytesIO

from typing import Optional
from typing import Tuple
from typing import List
from typing import Union

from molgraph.chemistry.ops import molecule_from_string


def visualize_maps(
    *,
    molecule: Union[str, Chem.Mol],
    maps: Union[List[float], Tuple[float, ...], np.ndarray],
    size: Union[None, str, Tuple[int, int]] = 'auto',
    padding: float = 0.1,
    save_path: Optional[str] = None,
) -> Optional[PngImageFile]:

    if not isinstance(molecule, Chem.Mol):
        molecule = molecule_from_string(molecule)

    if size == 'auto' or size is None:
        longest_path = Chem.GetDistanceMatrix(molecule).max()
        size = 120 + int(longest_path * 40)
        size = (size, size)

    drawer = Chem.Draw.MolDraw2DCairo(*size)
    options = drawer.drawOptions()
    options.padding = min(padding, 0.9)

    SimilarityMaps.GetStandardizedWeights(maps)
    SimilarityMaps.GetSimilarityMapFromWeights(
        mol=molecule,
        weights=[float(m) for m in maps],
        draw2d=drawer)

    drawer.FinishDrawing()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        drawer.WriteDrawingText(save_path)
        return

    return Image.open(BytesIO(drawer.GetDrawingText()))

def visualize_molecule(
    *,
    molecule: Union[str, Chem.Mol],
    atom_index: bool = False,
    bond_index: bool = False,
    size: Tuple[int, int] = (300, 300),
) -> Optional[PngImageFile]:
    if not isinstance(molecule, Chem.Mol):
        molecule = molecule_from_string(molecule)
    if atom_index:
        for atom in molecule.GetAtoms():
            atom.SetProp('atomNote', str(atom.GetIdx()))
    if bond_index:
        for bond in molecule.GetBonds():
            bond.SetProp('bondNote', str(bond.GetIdx()))
    return Chem.Draw.MolToImage(molecule, size=size)

def visualize_conformers(
    molecule: Chem.Mol
) -> Chem.Draw.IPythonConsole.display.Image:
    conformers = []
    energies = []
    for conformer in molecule.GetConformers():
        new_molecule = Chem.Mol(molecule)
        new_molecule.RemoveAllConformers()
        new_molecule.AddConformer(conformer, assignId=True)
        conformers.append(new_molecule)
        ff = Chem.rdForceFieldHelpers.UFFGetMoleculeForceField(
            new_molecule, confId=0)
        energies.append(ff.CalcEnergy())
    legends = [f"UFF energy = {energy:.2f}" for energy in energies]
    return Chem.Draw.IPythonConsole.ShowMols(
        conformers, legends=legends, subImgSize=(200, 200))
