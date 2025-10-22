# python3 setup.py bdist_wheel
# python3 -m twine upload dist/*
import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()


extras_require = {
        "parallel" : ['ray[default]', 'protobuf==3.20.2'],
                # protobuf req by ray :
                # https://github.com/ray-project/ray/issues/25205 ,
                # https://developers.google.com/protocol-buffers/docs/news/2022-05-06#python-updates
        "fftw"     : ['pyFFTW>=0.12.0'],
        # "symmetry" : [ 'sympy', 'spglib>=2', 'irrep>=2.3.3', 'spgrep'],
                }

extras_require["all"] = sum( (extras_require[k] for k in extras_require.keys()), [])
extras_require["default"] = sum( (extras_require[k] for k in extras_require.keys() if k != "fftw"), [])


setuptools.setup(
     name='wannierberri',
     author="Stepan S. Tsirkin",
     author_email="stepan.tsirkin@epfl.ch",
     description="Constructuion of Wannier functions and Wannier interpolation",
     long_description=long_description,
     long_description_content_type="text/markdown",
     python_requires='>=3.11',
     install_requires=[
                        'numpy>=2.0', 
                        'scipy>=1.13',
                        'colorama',
                        'termcolor',
                        'numba>=0.55.2',
                        'packaging>=20.8',
                        'fortio>=0.4',
                        'wannier90io',
                        'xmltodict',
                        'matplotlib',
                        'sympy', 
                        'spglib>=2', 
                        'irrep>=2.4.1', 
                        'spgrep'
                      ],
     extras_require=extras_require,
     url="https://wannier-berri.org",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
         "Operating System :: OS Independent",
                 ],
                )
