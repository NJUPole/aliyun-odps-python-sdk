#!/bin/bash
set -e -x

git clone https://github.com/NJUPole/aliyun-odps-python-sdk /io/odps
cd /io/aliyun-odps-python-sdk

# Compile wheels
for PYBIN in /opt/python/*/bin; do
	if [[ ${PYBIN} != *"26"* ]] && [[ ${PYBIN} != *"33"* ]] && [[ ${PYBIN} != *"34"* ]]; then
		"${PYBIN}/pip" install cython
		"${PYBIN}/pip" install -r /io/odps/requirements-full.txt
		"${PYBIN}/pip" wheel /io/odps -w wheelhouse/
done

for whl in wheelhouse/pyodps*.whl; do
    if [[ ${whl} == *"pyodps"* ]]; then
        auditwheel repair $whl -w /io/wheelhouse/
    fi
done

