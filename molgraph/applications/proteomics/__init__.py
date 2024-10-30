from molgraph.applications.proteomics.peptide import Peptide
from molgraph.applications.proteomics.peptide_encoders import PeptideGraphEncoder
from molgraph.applications.proteomics.peptide_models import PeptideModel

from molgraph.models.interpretability.activation_maps import GradientActivationMapping


PepSaliency = PeptideSaliency = GradientActivationMapping
PepGNN = PeptideGNN = PeptideModel