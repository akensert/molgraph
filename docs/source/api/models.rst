###########################
Models
###########################

***************************
Message passing
***************************

MPNN
===========================
.. autoclass:: molgraph.models.MPNN(tensorflow.keras.layers.Layer)
  :members: call, build, get_config, from_config
  :member-order: bysource

***************************
Interpretability
***************************

GradientActivationMapping
===========================
.. autoclass:: molgraph.models.GradientActivationMapping(tensorflow.keras.Model)
  :members: predict, get_config, from_config

SaliencyMapping
===========================
.. autoclass:: molgraph.models.SaliencyMapping(tensorflow.keras.Model)
  :members: predict, get_config, from_config

IntegratedSaliencyMapping
===========================
.. autoclass:: molgraph.models.IntegratedSaliencyMapping(tensorflow.keras.Model)
  :members: predict, get_config, from_config

SmoothGradSaliencyMapping
===========================
.. autoclass:: molgraph.models.SmoothGradSaliencyMapping(tensorflow.keras.Model)
  :members: predict, get_config, from_config
