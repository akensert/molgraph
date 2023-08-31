from molgraph.models.interpretability.activation_maps import GradientActivationMapping
from molgraph.models.interpretability.saliency import SaliencyMapping
from molgraph.models.interpretability.saliency import IntegratedSaliencyMapping
from molgraph.models.interpretability.saliency import SmoothGradSaliencyMapping
from molgraph.models.gin import GIN
from molgraph.models.mpnn import MPNN
from molgraph.models.dmpnn import DMPNN

from molgraph.models.pretraining.autoencoders import GraphAutoEncoder
from molgraph.models.pretraining.autoencoders import GraphVariationalAutoEncoder
from molgraph.models.pretraining.masked_modeling import GraphMasking

# aliases
GVAE = GraphVAE = GraphVariationalAutoEncoder 
GAE = GraphAE = GraphAutoEncoder
MaskedGraphModeling = GraphMasking

Saliency = VanillaSaliencyMapping = SaliencyMapping
IntegratedSaliency = IntegratedSaliencyMapping
SmoothGradSaliency = SmoothGradSaliencyMapping
GradCAM = GradientActivation = GradientActivationMapping
