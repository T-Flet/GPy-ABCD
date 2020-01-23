import io
import os
import re

from setuptools import find_packages, setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())


setup(
    name="GPy-ABCD",
    version="0.1.3",
    url="https://github.com/T-Flet/GPy-ABCD",
    license='BSD 3-Clause',

    author="Thomas Fletcher",
    author_email="T-Fletcher@outlook.com",

    description="Basic implementation with GPy of an Automatic Bayesian Covariance Discovery (ABCD) system",
    long_description=read("README.rst"),

    packages=find_packages(exclude=('tests',)),

    install_requires=[],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
    ],
)
