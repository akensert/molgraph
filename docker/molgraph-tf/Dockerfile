FROM tensorflow/tensorflow:2.9.1

FROM git clone https://github.com/akensert/molgraph.git && \
     cd molgraph && \
     pip install -e . && \
     rm -rf ~/.cache/pip && \
     rm -rf tensorflow-tutorials

WORKDIR /root/my-workspace
