# MolGraph docs

> [!NOTE]
> Documentation may not be up-to-date.

## Build the documentation

```
pip install -r requirements
make html
open build/html/index.html
```

## Test docstring examples

```
sphinx-build -M doctest "source" "build" source/api/layers.rst 
```
replace "layers.rst" with e.g. "models.rst" to test docstrings of models.
