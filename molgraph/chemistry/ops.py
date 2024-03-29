from rdkit import Chem

from typing import Union


def molecule_from_string(
    molecule: Union[str, Chem.Mol],
    catch_errors: bool = True,
) -> Union[Chem.Mol, None]:

    """Generates an RDKit molecule object from a SMILES string, InChI string,
    or SDF string. If an RDKit molecule object is inputted, it is immediately
    returned.
    """

    if isinstance(molecule, Chem.Mol):
        return molecule

    # Convert string to a RDKit molecule object (do not sanitize yet)
    if molecule.startswith('InChI'):
        molecule = Chem.MolFromInchi(molecule, sanitize=False)
    elif '\n' not in molecule:
        molecule = Chem.MolFromSmiles(molecule, sanitize=False)
    else:
        # assumed to be a block from an .sdf file
        molecule = molecule.rstrip('$$$$')
        molecule = Chem.MolFromMolBlock(molecule, sanitize=False)

    if molecule is None:
        # Inputted molecule was invalid
        raise ValueError(f"Inputted `molecule` ({molecule}) is invalid")

    # Sanitize the molecule, and catch errors (if `catch_errors` = True).
    # If an error occur, sanitize the molecule again, without the sanitization
    # step that caused the error previously. This should be used with caution,
    # as unrealistic molecules will go through without a raised error.
    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        if not catch_errors:
            return None
        Chem.SanitizeMol(
            molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^flag)

    # Assign stereo chemistry to molecule
    Chem.AssignStereochemistry(
        molecule, cleanIt=True, force=True, flagPossibleStereoCenters=True)

    # Return the (maybe partially) sanitized RDKit molecule object
    return molecule
