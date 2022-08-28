Installation
============

Install via **pip**:

.. code-block::

  pip install molgraph

Or via **docker**:

.. code-block::

  git clone https://github.com/akensert/molgraph.git
  cd molgraph/docker
  docker build -t molgraph-tf[-gpu][-jupyter]/molgraph:0.0 molgraph-tf[-gpu][-jupyter]/
  docker run -it [-p 8888:8888] molgraph-tf[-gpu][-jupyter]/molgraph:0.0
