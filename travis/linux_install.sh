docker pull $DOCKER_IMAGE
export PYVER=$PYVER
sudo chmod 777 travis/linux-wheel.sh
sudo chmod 777 travis/osx-wheel.sh
docker run --rm -e "PYVER=$PYVER" -v $pwd:/io $DOCKER_IMAGE $PRE_CMD /io/travis/linux-wheel.sh
