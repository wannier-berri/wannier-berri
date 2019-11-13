## python3 setup.py sdist
## python3 setup.py sdist bdist_wheel
## python -m twine upload *
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


from wannier19 import __version__ as version

setuptools.setup(
     name='wannier19',  
     version=version,
     author="Stepan S. Tsirkin",
     author_email="stepan.tsirkin@uzh.ch",
     description="Advanced tool for Wannier interpolation",
     long_description=long_description,
     long_description_content_type="text/markdown",
     install_requires=['numpy', 'scipy >= 1.0', 'lazy_property','colorama','termcolor','pyfiglet','termcolor'],
     url="https://github.com/stepan-tsirkin/wannier19",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
         "Operating System :: OS Independent",
     ],
 )