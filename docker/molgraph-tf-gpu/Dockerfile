FROM tensorflow/tensorflow:2.15.0-gpu

FROM git clone https://github.com/akensert/molgraph.git && \
     cd molgraph && \
     pip install -e .[gpu] && \
     rm -rf ~/.cache/pip && \
     rm -rf tensorflow-tutorials

WORKDIR /root/my-workspace
