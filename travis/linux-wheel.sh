#!/bin/bash
set -e -x

echo $(PYVER)

PYBIN=/opt/python/${PYVER}/bin
# Compile wheels
${PYBIN}/pip install -r ../requirements-full.txt
${PYBIN}/pip install cython
${PYBIN}/pip wheel -w wheelhouse/ .
