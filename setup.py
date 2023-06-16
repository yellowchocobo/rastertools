import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read_me(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "rastertools",
    version = "0.0.1",
    author = "Nils C. Prieur",
    author_email = "prieur.nils@gmail.com",
    description = ("Rastertools is a package that gathers a collection of tools to manipulate and extract metadata from rasters."),
    license = "MIT",
    keywords = "example documentation tutorial",
    url = "https://github.com/yellowchocobo/rastertools",
    packages=['rastertools'],
    long_description=read_me('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)
