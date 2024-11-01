import os
import sys
sys.path.insert(0, os.path.abspath('../package'))
autodoc_mock_imports = ["torch", "faiss"]

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Nearest Neighbor Normalization'
copyright = '2024, Sumedh Shenoy, Neil Chowdhury, Franklin Wang'
author = 'Sumedh Shenoy, Neil Chowdhury, Franklin Wang'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # for Google-style docstrings
    'sphinx_autodoc_typehints',  # for type hints in docs
    'sphinx.ext.autosummary',
]

autodoc_default_options = {
    'members': True,
    'special-members': '__init__'
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

html_css_files = [
    'custom.css',
]
