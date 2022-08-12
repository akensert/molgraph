from rdkit import Chem
from dataclasses import dataclass
from typing import Optional
from typing import List
from typing import Union

from molgraph.chemistry import conformer_utils
from molgraph.chemistry import transform_ops


@dataclass
class ConformerGenerator:

    num_conformer_candidates: Optional[Union[str, int]] = 'auto'
    embedding_method: str = 'ETKDGv2'
    force_field_method: Optional[str] = 'UFF'
    max_iter: Optional[Union[str, int]] = 'auto'
    keep_hydrogens: bool = False

    def __call__(self, molecule: Union[str, Chem.Mol]) -> Chem.Mol:

        if not isinstance(molecule, Chem.Mol):
            molecule = transform_ops.molecule_from_string(molecule)
            if molecule is None:
                return None

        molecule = Chem.AddHs(molecule)

        molecule = conformer_utils.embed_conformers(
            molecule, self.num_conformer_candidates, self.embedding_method)

        molecule = conformer_utils.force_field_minimize_conformers(
            molecule, self.force_field_method, self.max_iter)

        molecule = conformer_utils.get_lowest_energy_conformer(
            molecule, self.force_field_method)

        if not self.keep_hydrogens:
            molecule = Chem.RemoveHs(molecule)

        return molecule

    @property
    def available_embedding_methods(self) -> List[str]:
        return list(conformer_utils._embedding_method.keys())

    @property
    def available_force_field_methods(self) -> List[str]:
        return conformer_utils._available_ff_methods
