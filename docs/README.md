# wannier-berri.org
Web page of Wannier Berri. I (Stepan) regulary upload the master branch to the hosting server. 

Compling
---------

to generate the html just type "make"  and you will get the result in the `./html` folder, which you can check locally.

you need the sphinx installed, and prbably some other extensions

`pip3 install sphinx sphinx_pyreverse sphinx_sitemap sphinx_rtd_theme`

Autodoc
-------
Automated documentation will be generated for that wannierberri version which will be imported by the ``import wannierberri`` command. If you want to generate html for the local copy of the code, please make a symbolik link in the root of this repository via `ln -s /path/to/wannier-berri/wannierberri .`

Contributing
------------
If you read this, you are given also write acess to the repository, but master branch is protected. Please, create a branch and make a PR for any change.  After a PR is merged, I will compile on my machine and copy the contents of `./html` to the `~/public_html` folder on the hosting server.

PDF images
----------
html does not support `\*.pdf` figures, but all `\*.pdf` figures in `source/\*/` directories will be automatically converted  to `\*.pdf.svg`.  So in the .rst files please reference the `\*.pdf.svg` files instead of `pdf`s. Please do not add the converted figures to repository. If you add original svg figures, please do NOT name them `\*.pdf.svg` ,  otehwise they may be accidentally removed. To remove all converted figures type ``make clean_fig_svg``.  Youi need to have the `pdf2svg` utility installed

Table of contents
------------------

If you modify the structure of the site (change the titles of the sections, reorder sections, and new ones), 
then probably the table of contents is not updted on all pages. Therefor do  `make clean_html` before `make`

Privacy
-------
This is intended to be a private repository. Please do not fork it into your public repository, and avoid any other form of publoshing.
