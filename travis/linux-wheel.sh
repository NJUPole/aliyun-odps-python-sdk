#!/bin/bash
set -e -x
PYBIN=/opt/python/${PYVER}/bin
mkdir io/wheelhouse
# Compile wheels
${PYBIN}/pip install -r requirements-full.txt
${PYBIN}/pip install cython
${PYBIN}/pip wheel /io/ -w wheelhouse/ .
