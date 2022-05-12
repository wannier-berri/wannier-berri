## python3 setup.py bdist_wheel
## python3 -m twine upload dist/* 
import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='wannierberri',  
     author="Stepan S. Tsirkin",
     author_email="stepan.tsirkin@uzh.ch",
     description="Advanced tool for Wannier interpolation",
     long_description=long_description,
     long_description_content_type="text/markdown",
     python_requires='>=3.6',
     install_requires = [
                        'numpy',
                        'scipy >= 1.0', 
                        'lazy_property',
                        'colorama',
                        'termcolor',
                        'pyfiglet',
                        'numba',
                        'termcolor',
                        'pyFFTW>=0.12.0',
                        'packaging>=20.8',
                        'matplotlib',
                        'fortio>=0.4',
                        'ray[default]',
                        'sympy',
                        'spglib',
                        ],
     url="https://wannier-berri.org",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
         "Operating System :: OS Independent",
     ],
 )
