FROM tensorflow/tensorflow:2.9.1-gpu-jupyter

FROM git clone --depth 1 https://github.com/akensert/molgraph.git && \
     cd molgraph && \
     pip install . && \
     rm -rf ~/.cache/pip && \
     rm -rf tensorflow-tutorials

WORKDIR /root/my-workspace
