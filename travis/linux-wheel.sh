#!/bin/bash
set -e -x
PYBIN=/opt/python/${PYVER}/bin
${PYBIN}/pip install --disable-pip-version-check --user --upgrade pip
${PYBIN}/pip install cython
cd /io/
mkdir wheelhouse
# Compile wheels
${PYBIN}/python setup.py build
${PYBIN}/python setup.py bdist_wheel
