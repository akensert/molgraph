Welcome to MolGraph!
=======================

*This is an early release; things are still being updated and added. Hence, API compatibility may break in the future.*

**What is MolGraph?** A light-weight Python package for applying graph neural networks (GNNs) on molecular graphs.
It is built with, and aims to be highly compatible with,
`TensorFlow <https://www.tensorflow.org/api_docs/python/tf>`_ and
`Keras <https://keras.io/>`_.

**Why MolGraph?** As it integrates well with TensorFlow and Keras, it allows for easy and flexible
implementations of GNNs. Furthermore, the focus is specifically, and exclusively,
on small molecules, with a dedicated chemistry module for customizing the molecular
graph.


.. toctree::
  :glob:
  :maxdepth: 1
  :caption: Get started

  get_started/installation
  get_started/walk_through

.. toctree::
  :glob:
  :maxdepth: 1
  :caption: API

  api/tensors
  api/chemistry
  api/layers
  api/models


.. toctree::
  :glob:
  :maxdepth: 1
  :caption: Tutorials

  examples/tutorials/*
