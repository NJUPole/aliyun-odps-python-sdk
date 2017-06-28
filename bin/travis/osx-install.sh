brew update
# Per the `pyenv homebrew recommendations <https://github.com/yyuu/pyenv/wiki#suggested-build-environment>`_.
brew install openssl readline
# See https://docs.travis-ci.com/user/osx-ci-environment/#A-note-on-upgrading-packages.
# I didn't do this above because it works and I'm lazy.
    # install pyenv
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
PYENV_ROOT="$HOME/.pyenv"
PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
pyenv install $PYTHON
pyenv global $PYTHON
python -V
export CC=clang
export CXX=clang++
pip install --disable-pip-version-check --user --upgrade pip
pip install cython
pip install wheel
# wheel
#python setup.py bdist_wheel
pip wheel --no-deps .
#repair_wheel
mkdir dist
cp *.whl dist/
#pip install delocate
delocate-wheel dist/*.whl
#delocate-addplat --rm-orig -x 10_9 dist/*.whl
cd dist
ls dist
pip install pyodps*.whl
python -c "from odps.tunnel.checksum_c import Checksum"