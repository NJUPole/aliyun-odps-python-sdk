#!/bin/bash
# Depends on:
#   BUILD_COMMANDS
#   PYTHON_VERSION
#   BUILD_COMMIT (may be used by config.sh)
#   UNICODE_WIDTH  (can be empty)
#   BUILD_DEPENDS  (may be used by config.sh, can be empty)
set -e

# Unicode width, default 32
UNICODE_WIDTH=${UNICODE_WIDTH:-32}

# Location of wheels, default "wheelhouse"
WHEEL_SDIR=${WHEEL_SDIR:-wheelhouse}

# Always pull in common and library builder utils
MULTIBUILD_DIR=$(dirname "${BASH_SOURCE[0]}")
# These routines also source common_utils.sh
echo "OK1"
source $MULTIBUILD_DIR/manylinux_utils.sh
echo "OK2"
source $MULTIBUILD_DIR/library_builders.sh
echo "OK3"
# Set PATH for chosen Python, Unicode width
export PATH="$(cpython_path $PYTHON_VERSION $UNICODE_WIDTH)/bin:$PATH"

# Change into root directory of repo
cd /io/$REPO_NAME

# Configuration for this package, possibly overriding `build_wheel` defined in
# `common_utils.sh` via `manylinux_utils.sh`.
source config.sh
echo "OK4"
$BUILD_COMMANDS