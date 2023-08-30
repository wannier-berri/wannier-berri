## python3 setup.py bdist_wheel
## python3 -m twine upload dist/*
import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='wannierberri',
     author="Stepan S. Tsirkin",
     author_email="stepan.tsirkin@ehu.eus",
     description="Advanced tool for Wannier interpolation",
     long_description=long_description,
     long_description_content_type="text/markdown",
     python_requires='>=3.7',
     install_requires=[
                        'numpy<1.25,>=1.24', # reauired by numba . min 1.24 required by scipy.io
                        'scipy>=1.0',
                        'lazy_property',
                        'colorama',
                        'termcolor',
                        'pyfiglet',
                        'numba>=0.55.2',
                        'termcolor',
                        'pyFFTW>=0.12.0',
                        'packaging>=20.8',
                        'matplotlib',
                        'fortio>=0.4',
                        'protobuf==3.20.2', # req by ray : https://github.com/ray-project/ray/issues/25205 , https://developers.google.com/protocol-buffers/docs/news/2022-05-06#python-updates
                        'ray[default]',
                        'sympy',
                        'spglib<2', # TODO : make compatible with latest version
                        'untangle',
                        'irrep>=1.8.2',
                      ],
     url="https://wannier-berri.org",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
         "Operating System :: OS Independent",
                 ],
                )
