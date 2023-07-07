from molgraph.models.interpretability.activation_maps import GradientActivationMapping
from molgraph.models.interpretability.saliency import SaliencyMapping
from molgraph.models.interpretability.saliency import IntegratedSaliencyMapping
from molgraph.models.interpretability.saliency import SmoothGradSaliencyMapping
from molgraph.models.mpnn import MPNN
from molgraph.models.dmpnn import DMPNN
from molgraph.models.dgin import DGIN

from molgraph.models.pretraining.autoencoders import GraphAutoEncoder
from molgraph.models.pretraining.autoencoders import GraphVariationalAutoEncoder

# aliases
VanillaSaliencyMapping = SaliencyMapping
GVAE = GraphVAE = GraphVariationalAutoEncoder 
GAE = GraphAE = GraphAutoEncoder

Saliency = SaliencyMapping
IntegratedSaliency = IntegratedSaliencyMapping
SmoothGradSaliency = SmoothGradSaliencyMapping
GradCAM = GradientActivation = GradientActivationMapping