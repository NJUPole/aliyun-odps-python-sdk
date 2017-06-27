#!/bin/bash
set -e -x
PYBIN=/opt/python/${PYVER}/bin
# Compile wheels
${PYBIN}/pip install -r /io/requirements-full.txt
${PYBIN}/pip install cython
${PYBIN}/pip wheel -w wheelhouse/ .
