set -e

brew update
# Per the `pyenv homebrew recommendations <https://github.com/yyuu/pyenv/wiki#suggested-build-environment>`_.
brew upgrade openssl@1.1 readline
CFLAGS="-I/opt/local/include/"
LDFLAGS="-L/opt/local/lib/"
# See https://docs.travis-ci.com/user/osx-ci-environment/#A-note-on-upgrading-packages.
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
PYENV_ROOT="$HOME/.pyenv"
PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"


pyenv install $PYTHON
pyenv global $PYTHON


#check python version
python -V
pip install --upgrade pip
pip install cython wheel numpy

pip wheel --no-deps .
#repair_wheel
mkdir dist
cp *.whl dist/
pip install delocate
delocate-wheel dist/*.whl
delocate-addplat --rm-orig -x 10_9 -x 10_10 dist/*.whl

