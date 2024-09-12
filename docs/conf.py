# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Platypus-Opt'
copyright = '2015-2024, David Hadka'
author = 'David Hadka'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.imgmath',
    'sphinx_rtd_theme',
]

templates_path = ['_templates']
exclude_patterns = []

language = 'en'

add_module_names = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# The initial api/ contents were created with:
#   sphinx-apidoc --separate --remove-old --no-toc -o docs/api platypus "*test*"
#
# Build these docs locally with:
#   sphinx-build -M html . _build
