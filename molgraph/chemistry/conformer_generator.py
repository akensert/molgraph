from rdkit import Chem

from dataclasses import dataclass

from typing import Optional
from typing import List
from typing import Union

from molgraph.chemistry import conformer_utils
from molgraph.chemistry.ops import molecule_from_string


@dataclass
class ConformerGenerator:

    '''Conformer generator to generate molecular conformers.

    Args:
        num_conformer_candidates (int, str, None):
            Number of conformers to generate, from which the conformer with
            the lowest energy will be selected. If 'auto', the number of
            conformers will depend on the size of the molecule. Default to
            'auto'.
        embedding_method (str):
            The embedding method to use. Either of 'ETDG', 'ETKDG', 'ETKDGv2',
            'ETKDGv3', 'srETKDGv3' or 'KDG'. Default to 'ETKDGv2'.
        force_field_method (str):
            The force field method to use. Either of 'MMFF', 'MMFF94', 'MMFF94s'
            or 'UFF'. Default to 'UFF'.
        max_iter (int, str, None):
            Maximum number of iterations for generating a conformer. If 'auto',
            the number of iterations will depend on the size of the molecule.
            Default to 'auto'.
        keep_hydrogens (bool):
            Whether to keep the hydrogens of the selected conformer. Default
            to False.
    '''

    num_conformer_candidates: Optional[Union[str, int]] = 'auto'
    embedding_method: str = 'ETKDGv2'
    force_field_method: Optional[str] = 'UFF'
    max_iter: Optional[Union[str, int]] = 'auto'
    keep_hydrogens: bool = False

    def __call__(self, molecule: Union[str, Chem.Mol]) -> Chem.Mol:

        if not isinstance(molecule, Chem.Mol):
            molecule = molecule_from_string(molecule)
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
        'Available embedding methods for conformer generation.'
        return list(conformer_utils._embedding_method.keys())

    @property
    def available_force_field_methods(self) -> List[str]:
        'Available force field methods for conformer optimization.'
        return conformer_utils._available_ff_methods
