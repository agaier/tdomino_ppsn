import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="bbq",

    description="Tasty add-ons for pyRibs.",

    author="Adam Gaier",

    packages=find_packages(exclude=['data', 'figures', 'output', 'notebooks']),

    version="0.0.2",
)
