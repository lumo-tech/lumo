# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import re
from pathlib import Path

version_fn = Path(__file__).parent.parent.parent.joinpath('src/lumo/__init__.py').as_posix()


def extract_version():
    return re.search(
        r'__version__ = "([\d.d\-]+)"',
        open(version_fn, 'r', encoding='utf-8').read()).group(1)


project = 'lumo'
copyright = '2023, sailist'
author = 'sailist'
release = extract_version()

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath('../../src/'))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

from recommonmark.transform import AutoStructify
from recommonmark.parser import CommonMarkParser
import commonmark

extensions = [
    "myst_parser",
    # "autodoc2",
    "sphinx_rtd_theme",  # pip install sphinx_rtd_theme
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',

]


def docstring(app, what, name, obj, options, lines):
    md = '\n'.join(lines)
    ast = commonmark.Parser().parse(md)
    rst = commonmark.ReStructuredTextRenderer().render(ast)
    lines.clear()
    lines += rst.splitlines()


def setup(app):
    app.add_transform(AutoStructify)
    app.add_config_value('recommonmark_config', {
        'auto_toc_tree_section': 'Contents',
    }, True)
    app.connect('autodoc-process-docstring', docstring)


autodoc2_packages = [
    "../../src/library",
]
myst_enable_extensions = [
    "deflist",
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
#
# source_suffix = {
#     '.rst': 'restructuredtext',
#     '.txt': 'markdown',
#     '.md': 'markdown',
# }
#
commonmark_suffixes = ['.rst']

source_parsers = {
    '.md': CommonMarkParser,
}

source_suffix = ['.rst', '.md']

autodoc2_docstring_parser_regexes = [
    # this will render all docstrings as Markdown
    (r".*", "myst"),
    # this will render select docstrings as Markdown
    (r"autodoc2\..*", "myst"),
]
