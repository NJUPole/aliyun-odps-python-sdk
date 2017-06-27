#!/bin/bash
set -e -x

echo $(PYVER)

PYBIN=/opt/python/${PYVER}/bin
# Compile wheels
${PYBIN}/pip install -r requirements-full.txt
${PYBIN}/pip install cython
${PYBIN}/pip wheel -w wheelhouse/ .


# Install packages and test
for PYBIN in /opt/python/*/bin/; do
    "${PYBIN}/pip" install python-manylinux-demo --no-index -f /io/wheelhouse
    (cd "$HOME"; "${PYBIN}/nosetests" pymanylinuxdemo)
done