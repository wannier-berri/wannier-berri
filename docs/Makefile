# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = ./

default: html


# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


.PHONY: help Makefile



fig_subdirs := $(wildcard source/imag/*/ )
fig_sources := $(wildcard $(addsuffix *.pdf,$(fig_subdirs)))
fig_objects := $(patsubst %.pdf,%.pdf.svg,$(fig_sources))

figures : $(fig_objects)

$(fig_objects) : %.pdf.svg : %.pdf

%.pdf.svg : %.pdf
	pdf2svg $(CONVERTFLAGS) $< $@

html : figures

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
html : Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean_fig_svg :
	rm source/imag/*/*.pdf.svg

clean_html :
	rm html/*.html
	rm -r doctrees

