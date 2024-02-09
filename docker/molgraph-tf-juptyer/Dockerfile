FROM tensorflow/tensorflow:2.15.0-jupyter

FROM git clone --depth 1 https://github.com/akensert/molgraph.git && \
     cd molgraph && \
     pip install . && \
     rm -rf ~/.cache/pip && \
     rm -rf tensorflow-tutorials

WORKDIR /root/my-workspace
