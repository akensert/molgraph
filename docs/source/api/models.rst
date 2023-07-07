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

DMPNN
===========================
.. autoclass:: molgraph.models.DMPNN(tensorflow.keras.layers.Layer)
  :members: call, build, get_config, from_config
  :member-order: bysource

DGIN
===========================
.. autoclass:: molgraph.models.DGIN(tensorflow.keras.layers.Layer)
  :members: call, build, get_config, from_config
  :member-order: bysource

***************************
Interpretability
***************************

GradientActivationMapping
===========================
.. autoclass:: molgraph.models.GradientActivationMapping(tensorflow.Module)
  :members: build
  :special-members: __call__

SaliencyMapping
===========================
.. autoclass:: molgraph.models.SaliencyMapping(tensorflow.Module)
  :members: build
  :special-members: __call__

IntegratedSaliencyMapping
===========================
.. autoclass:: molgraph.models.IntegratedSaliencyMapping(tensorflow.Module)
  :members: build
  :special-members: __call__

SmoothGradSaliencyMapping
===========================
.. autoclass:: molgraph.models.SmoothGradSaliencyMapping(tensorflow.Module)
  :members: build
  :special-members: __call__


***************************
Pretraining
***************************

GraphAutoEncoder
===========================
.. autoclass:: molgraph.models.GraphAutoEncoder(tensorflow.keras.Model)
  :members: fit, evaluate, predict, get_config, from_config

GraphVariationalAutoEncoder
===========================
.. autoclass:: molgraph.models.GraphVariationalAutoEncoder(tensorflow.keras.Model)
  :members: fit, evaluate, predict, get_config, from_config

