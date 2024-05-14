# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
import inspect

import molgraph

doctest_global_setup = '''import molgraph
import tensorflow as tf
import rdkit

tf.config.set_visible_devices([], 'GPU')'''

# -- Project information -----------------------------------------------------

project = 'MolGraph'
copyright = '2022, Alexander Kensert'
author = 'Alexander Kensert'

# The full version, including alpha/beta/rc tags
release = '0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.doctest',
    'sphinx.ext.autosummary',
    'sphinx_gallery.load_style',
    'sphinx.ext.linkcode',
    'nbsphinx'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['../_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

nbsphinx_allow_errors = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_logo = '_static/molgraph-logo3.png'

html_theme_options = {}

nbsphinx_thumbnails = {} # .py : .png

autodoc_typehints = 'none'

autodoc_default_options = {
    'member-order': 'groupwise',
}

def linkcode_resolve(domain, info):

    if domain != 'py':
        return None

    modulename = info['module']
    fullname = info['fullname']

    module = sys.modules.get(modulename)

    if module is None:
        return None

    for part in fullname.split('.'):
        try:
            module = getattr(module, part)
        except Exception:
            return None
    try:
        filepath = inspect.getsourcefile(module)
    except Exception:
        filepath = None
    if not filepath:
        return None

    try:
        source_code, line_number = inspect.getsourcelines(module)
    except Exception:
        line_number = None

    filepath = os.path.relpath(
        filepath, start=os.path.dirname(molgraph.__file__))

    url = 'https://github.com/akensert/molgraph/tree/main/molgraph/'
    url += filepath
    if line_number:
        return url + f'#L{line_number}-L{line_number + len(source_code)- 1}'
    return url
