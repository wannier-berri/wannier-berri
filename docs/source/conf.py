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
sys.path.insert(0, os.path.abspath('../'))

import wannierberri

# -- Project information -----------------------------------------------------

project = 'Wannier Berri'
copyright = '2021, Stepan Tsirkin' 
author = 'Stepan Tsirkin'
numfig = True
master_doc = 'index'

# The full version, including alpha/beta/rc tags
release = wannierberri.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
#extensions = ['sphinx.ext.autodoc']

extensions = [
    'sphinx.ext.autodoc', 'sphinx.ext.mathjax', 'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode', 'sphinx_pyreverse' , 'sphinx_sitemap' , 'sphinx.ext.napoleon',
    'nbsphinx',
]

napoleon_custom_sections = ["Sets"]
nbsphinx_allow_errors = True

html_baseurl = 'https://docs.wannier-berri.org'
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'alabaster'
#html_theme = 'groundwork'
#html_theme = 'basic'

#html_theme = 'sphinx_drove_theme'
#import sphinx_drove_theme
#html_theme_path = [sphinx_drove_theme.get_html_theme_path()]

#import sphinx_pdj_theme
#html_theme = 'sphinx_pdj_theme'
#htm_theme_path = [sphinx_pdj_theme.get_html_theme_path()]

# sets the darker appearence
#html_theme_options = {     'style': 'darker' }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
def setup(app):
    app.add_css_file('css/custom.css')


# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#

#html_sidebars = {
#    '**': ['globaltoc.html', 'localtoc.html']
#}

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
html_use_opensearch = 'https://docs.wannier-berri.org'

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}
#---sphinx-themes-----
#html_favicon = 'favicon.ico'
#html_theme = 'sphinx_pdj_theme'
#import sphinx_pdj_theme
#html_theme_path = [sphinx_pdj_theme.get_html_theme_path()]

html_theme = 'sphinx_rtd_theme'
html_favicon = 'imag/logo-WB/WB-logo.ico'
#html_logo = 'imag/logo-WB/WANNIERBERRI-redblack.png'
html_logo = 'imag/logo-WB/Book.png'
html_show_sourcelink = False

# True Basque colors
bred='#D62618' 
bgreen= '#009C46'


red='#a00000'
green='#008880'

html_theme_options = {
    'canonical_url': '',
#    'analytics_id': 'UA-XXXXXXX-1',  #  Provided by Google in your dashboard
    'logo_only': True,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
#    'vcs_pageview_mode': '',
    'style_nav_header_background':  bgreen , # '#009C46', #  'white',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

def setup(app):
     app.add_css_file('style.css')