import numpy as np
import os
from rdkit import Chem
from rdkit.Chem.Draw import SimilarityMaps
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import List
from typing import Dict
import IPython

from molgraph.chemistry import transform_ops


def visualize_maps(
    molecule: Chem.Mol,
    maps: np.ndarray,
    save_path: Optional[str] = None,
    size: Tuple[int, int] = (300, 300)
) -> None:

    if isinstance(molecule, str):
        molecule = transform_ops.molecule_from_string(molecule)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        drawer = Chem.Draw.MolDraw2DCairo(*size)
        drawer.drawOptions().bondLineWidth = 3
    else:
        drawer = None

    if maps is not None:
        maps = maps / np.max(np.abs(maps))
    else:
        maps = np.zeros(molecule.GetNumAtoms(), dtype='float32')

    SimilarityMaps.GetSimilarityMapFromWeights(
        mol=molecule,
        weights=[float(m) for m in maps],
        size=size,
        coordScale=1.0,
        colors='g',
        alpha=0.4,
        contourLines=10,
        draw2d=drawer)


    if save_path:
        drawer.FinishDrawing()
        drawer.WriteDrawingText(save_path)


def visualize_conformers(molecule: Chem.Mol) -> IPython.core.display.Image:
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
