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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import sys
import os
from subprocess import run, CalledProcessError
import inspect
import pkg_resources

from pybtex.style.formatting.plain import Style as PlainStyle
from pybtex.style.labels import BaseLabelStyle
from pybtex.plugin import register_plugin


# -- Project information -----------------------------------------------------
project = 'zennit'
copyright = '2021, chr5tphr'
author = 'chr5tphr'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.linkcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.extlinks',
    'sphinx_rtd_theme',
    'sphinx_copybutton',
    'sphinxcontrib.datatemplates',
    'sphinxcontrib.bibtex',
    'nbsphinx',
]


def config_inited_handler(app, config):
    os.makedirs(os.path.join(app.srcdir, app.config.generated_path), exist_ok=True)


def setup(app):
    app.add_config_value('REVISION', 'master', 'env')
    app.add_config_value('generated_path', '_generated', 'env')
    app.connect('config-inited', config_inited_handler)


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# interactive badges for binder and colab
nbsphinx_prolog = r"""
{% set docname = 'docs/source/' + env.doc2path(env.docname, base=False) %}

.. raw:: html

    <div class="admonition note">
      This page was generated from
      <a class="reference external" href="https://github.com/chr5tphr/zennit/blob/{{ env.config.REVISION }}/{{ docname|e }}">{{ docname|e }}</a>
      <br />
      Interactive online version:
      <span style="white-space: nowrap;">
        <a href="https://mybinder.org/v2/gh/chr5tphr/zennit/{{ env.config.REVISION|e }}?filepath={{ docname|e }}">
            <img alt="launch binder" src="https://mybinder.org/badge_logo.svg" style="vertical-align:text-bottom">
        </a>
      </span>
      <span style="white-space: nowrap;">
        <a href="https://colab.research.google.com/github/chr5tphr/zennit/blob/{{ env.config.REVISION|e }}/{{ docname|e }}">
            <img alt="Open in Colab" src="https://colab.research.google.com/assets/colab-badge.svg" style="vertical-align:text-bottom">
        </a>
      </span>
    </div>
"""

# autosummary_generate = True

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_line_continuation_character = "\\"
copybutton_here_doc_delimiter = "EOT"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
# html_theme = 'alabaster'

html_favicon = '_static/favicon.svg'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

bibtex_bibfiles = ['bibliography.bib']
bibtex_default_style = 'author_year_style'
bibtex_reference_style = 'author_year'


class AuthorYearLabelStyle(BaseLabelStyle):
    def format_labels(self, sorted_entries):
        for entry in sorted_entries:
            yield f'[{entry.persons["author"][0].last_names[0]} et al., {entry.fields["year"]}]'


class AuthorYearStyle(PlainStyle):
    default_label_style = AuthorYearLabelStyle


register_plugin('pybtex.style.formatting', 'author_year_style', AuthorYearStyle)


def getrev():
    try:
        revision = run(
            ['git', 'describe', '--tags', 'HEAD'],
            capture_output=True,
            check=True,
            text=True
        ).stdout[:-1]
    except CalledProcessError:
        revision = 'master'

    return revision


REVISION = getrev()

extlinks = {
    'repo': (
        f'https://github.com/chr5tphr/zennit/blob/{REVISION}/%s',
        '%s'
    )
}

LINKCODE_URL = (
    f'https://github.com/chr5tphr/zennit/blob/{REVISION}'
    '/src/{filepath}#L{linestart}-L{linestop}'
)


# revised from https://gist.github.com/nlgranger/55ff2e7ff10c280731348a16d569cb73
def linkcode_resolve(domain, info):
    if domain != 'py' or not info['module']:
        return None

    modname = info['module']
    topmodulename = modname.split('.')[0]
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    try:
        modpath = pkg_resources.require(topmodulename)[0].location
        filepath = os.path.relpath(inspect.getsourcefile(obj), modpath)
        if filepath is None:
            return
    except Exception:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        return None
    else:
        linestart, linestop = lineno, lineno + len(source) - 1

    return LINKCODE_URL.format(filepath=filepath, linestart=linestart, linestop=linestop)
