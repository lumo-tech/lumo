# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import re
from pathlib import Path

from sphinx.application import Sphinx

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

    "sphinx_multiversion",
    # "custom_rtd_theme",  # pip install custom_rtd_theme
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',

]


def docstring(app, what, name, obj, options, lines):
    md = '\n'.join(lines)
    ast = commonmark.Parser().parse(md)
    rst = commonmark.ReStructuredTextRenderer().render(ast)
    lines.clear()
    lines += rst.splitlines()


def html_page_context(app, pagename, templatename, context, doctree):
    # versioninfo = VersionInfo(
    #     app, context, app.config.smv_metadata, app.config.smv_current_version
    # )
    context["READTHEDOCS"] = True
    context["theme_display_version"] = True
    # context["vhasdoc"] = versioninfo.vhasdoc
    # context["vpathto"] = versioninfo.vpathto
    #
    context["current_version"] = extract_version()
    # context["latest_version"] = versioninfo[app.config.smv_latest_version]
    # context["html_theme"] = app.config.html_theme
    pass


def setup(app: Sphinx):
    app.add_transform(AutoStructify)
    app.add_config_value('recommonmark_config', {
        'auto_toc_tree_section': 'Contents',
    }, True)
    app.connect('autodoc-process-docstring', docstring)
    app.connect("html-page-context", html_page_context)

    app.add_html_theme('custom_rtd_theme', Path(__file__).parent.joinpath('custom_rtd_theme/').as_posix())


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


# html_theme = Path(__file__).parent.joinpath('custom_rtd_theme/').as_posix()
html_theme = "custom_rtd_theme"
# html_theme = "sphinx_rtd_theme"
# html_theme = "alabaster"
html_theme_options = {
    # 'github_url': 'https://github.com/pytorch-lumo/lumo',
    # 'analytics_id': 'G-XXXXXXXXXX',  #  Provided by Google in your dashboard
    # 'analytics_anonymize_ip': False,
    # 'logo_only': False,
    # 'display_version': True,
    # 'prev_next_buttons_location': 'bottom',
    # 'style_external_links': False,
    # 'vcs_pageview_mode': '',
    # 'style_nav_header_background': 'white',
    # # Toc options
    # 'collapse_navigation': True,
    # 'sticky_navigation': True,
    # 'navigation_depth': 4,
    # 'includehidden': True,
    # 'titles_only': False
}

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

# html_sidebars = {
#     '**': [
#         # "my_side.html",
#         # 'versions.html',
#     ],
# }
smv_outputdir_format = f'v{extract_version()}'
# smv_tag_whitelist = r'^.*$'  # Include all tags
smv_tag_whitelist = r'^v\d+\.\d+.\d+.\d+$'  # Include tags like "v2.1"
