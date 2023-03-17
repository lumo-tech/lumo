# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import recommonmark
from recommonmark.transform import AutoStructify
from recommonmark.parser import CommonMarkParser


def setup(app):
    # app.add_config_value('recommonmark_config', {
    #     'auto_toc_tree_section': 'Contents',
    # }, True)
    app.add_transform(AutoStructify)


project = 'lumo'
copyright = '2023, sailist'
author = 'sailist'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    "myst_parser",

]

myst_enable_extensions = [
    "deflist",
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
#
# source_suffix = {
#     '.rst': 'restructuredtext',
#     '.txt': 'markdown',
#     '.md': 'markdown',
# }
#

source_parsers = {
    '.md': CommonMarkParser,
}

source_suffix = ['.rst', '.md']
