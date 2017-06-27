#!/bin/bash
set -e -x
PYBIN=/opt/python/${PYVER}/bin
mkdir io/wheelhouse
# Compile wheels
${PYBIN}/pip install -r /io/requirements-full.txt
${PYBIN}/pip install cython
${PYBIN}/python /io/setup.py build
${PYBIN}/python /io/setup.py bdist_wheel
