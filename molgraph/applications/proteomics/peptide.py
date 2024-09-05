import re
import functools
from rdkit import Chem

from molgraph.applications.proteomics.definitions import _residue_smiles 
from molgraph.applications.proteomics.definitions import _residue_index



class Peptide:

    def __init__(self, sequence: str) -> None:
        self._sequence = sequence
        self._split_sequence = _sequence_split(self._sequence) 
        self._num_residues = len(self._split_sequence)
        _check_canonical(self)

    def __repr__(self) -> str:
        return f"<Peptide: {self._sequence!r} at {hex(id(self))}>"
    
    def __len__(self) -> int:
        return self._num_residues
    
    def __iter__(self) -> 'Peptide':
        self._i = 0
        return self
    
    def __next__(self) -> str:
        if self._i < self._num_residues:
            residue = self._split_sequence[self._i]
            self._i += 1 
            return residue 
        else:
            raise StopIteration

    @property
    def smiles(self) -> str:
        smiles_list = [_residue_smiles[residue] for residue in self]
        return _concatenate_smiles(smiles_list)

    @property
    def residue_sizes(self) -> list[int]:
        sizes = []
        for i, residue in enumerate(self):
            size = _num_atoms(_residue_smiles[residue])
            last_residue = i == (len(self) - 1)
            if not last_residue:
                size -= 1
            sizes.append(size)
        return sizes
    
    @property
    def residue_indices(self) -> list[int]:
        return [_extract_residue_index(residue) for residue in self]


@functools.lru_cache(maxsize=4096)
def _num_atoms(smiles: str) -> int:
    return Chem.MolFromSmiles(smiles).GetNumAtoms()

# TODO: Raise warning and canonicalize 
def _check_canonical(peptide: Peptide) -> None:
    for residue in peptide:
        smiles: str = _residue_smiles[residue]
        if not smiles.startswith('N') or not smiles.endswith('(=O)O'):
            raise ValueError(
                f"AA SMILES string {smiles!r} needs to be canonical: "
                "starting with 'N' and ending with '(=O)O'."
            )

def _sequence_split(sequence: str) -> list[str]:
    patterns = [
        r'(\[[A-Za-z0-9]+\]-[A-Z]\[[A-Za-z0-9]+\])', # N-term mod + mod
        r'([A-Z]\[[A-Za-z0-9]+\]-\[[A-Za-z0-9]+\])', # C-term mod + mod
        r'([A-Z]-\[[A-Za-z0-9]+\])', # C-term mod
        r'(\[[A-Za-z0-9]+\]-[A-Z])', # N-term mod
        r'([A-Z]\[[A-Za-z0-9]+\])', # Mod
        r'([A-Z])', # No mod
    ]
    return [match.group(0) for match in re.finditer("|".join(patterns), sequence)]

def _concatenate_smiles(smiles_list: list[str], sep: str = '') -> str:
    # ['NCC(=O)O', 'NCC(=O)O', ...] -> 'NCC(=O)NCC(=O)...'
    smiles_list = [
        smiles.rstrip("O") if i < len(smiles_list) - 1 else smiles
        for (i, smiles) in enumerate(smiles_list)
    ]
    return sep.join(smiles_list)

def _extract_residue_type(residue_tag: str) -> str:
    pattern = r"(?<!\[)[A-Z](?![\w-])"
    return [match.group(0) for match in re.finditer(pattern, residue_tag)][0]

def _extract_residue_index(residue_tag: str) -> int:
    return _residue_index[_extract_residue_type(residue_tag)]
