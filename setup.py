import setuptools
import os
import sys

def get_version():
  version_path = os.path.join(os.path.dirname(__file__), 'molgraph')
  sys.path.insert(0, version_path)
  from _version import __version__ as version
  return version

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
    "tensorflow==2.15.*",
    "rdkit==2023.9.5",
    "pandas>=1.0.3",
    "ipython==8.12.0",
]
extras_require = {
   'gpu': ['tensorflow[and-cuda]==2.15.*']
}

setuptools.setup(
    name='molgraph',
    version=get_version(),
    author="Alexander Kensert",
    author_email="alexander.kensert@gmail.com",
    description="Graph Neural Networks for Molecular Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/akensert/molgraph",
    packages=setuptools.find_packages(include=["molgraph*"]),
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS"
    ],
    python_requires=">=3.10",
    keywords=[
        'machine-learning',
        'deep-learning',
        'graph-neural-networks',
        'graphs',
        'molecular-machine-learning',
        'molecular-graphs',
        'cheminformatics',
        'chemometrics',
        'bioinformatics',
        'chemistry',
        'biology',
        'biochemistry',
    ]
)
