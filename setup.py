import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
    "tensorflow>=2.7.0",
    "numpy>=1.21.2",
    "rdkit-pypi>=2021.3.4",
    "pandas>=1.0.3"
]
tests_require = [
    "pytest>=7.1.2"
]
setuptools.setup(
    name='molgraph',
    version="0.0",
    author="Alexander Kensert",
    author_email="alexander.kensert@gmail.com",
    description="Implementations of graph neural networks for molecular machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/akensert/molgraph",
    packages=setuptools.find_packages(include=["molgraph*"]),
    install_requires=install_requires,
    tests_require=tests_require,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research"
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS"
    ],
    python_requires=">=3.6",
    keywords=[
        'molgraph',
        'chemistry',
        'cheminformatics',
        'chemoinformatics'
        'chemometrics',
        'bioinformatics',
        'machine-learning',
        'molecular-machine-learning',
    ]
)
