## Install MolGraph with Docker

<pre>
git clone https://github.com/akensert/molgraph.git
cd molgraph/docker
docker build -t molgraph-tf[-jupyter][-gpu]/molgraph:0.0 molgraph-tf[-jupyter][-gpu]/
docker run -it <b>[-p 8888:8888]</b> molgraph-tf<b>[-jupyter]</b>[-gpu]/molgraph:0.0
</pre>