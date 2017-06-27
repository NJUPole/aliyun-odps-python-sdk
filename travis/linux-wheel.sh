#!/bin/bash
set -e -x
PYBIN=/opt/python/${PYVER}/bin
${PYBIN}/pip install --disable-pip-version-check --user --upgrade pip
${PYBIN}/pip install cython
mkdir io/wheelhouse
# Compile wheels
${PYBIN}/pip install -r /io/requirements-full.txt
${PYBIN}/python /io/setup.py build
${PYBIN}/python /io/setup.py bdist_wheel
