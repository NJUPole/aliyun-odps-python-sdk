docker pull $DOCKER_IMAGE
export PYVER=$PYVER
docker run --rm -e "PYVER=$PYVER" -v `pwd`:/io $DOCKER_IMAGE /io/travis/linux-wheel.sh
