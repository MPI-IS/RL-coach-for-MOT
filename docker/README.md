# Container Images

In this directory is a Dockerfile that can be used to build a containers that has coach and everything installed/included in order to run a training on a MOT simulator.  How to build and use the container is defined below:

## default `Dockerfile`
* `make build_base` to create the image
* will create a basic Coach installation along with the continuous control MOT environment.
* `make shell_base` will launch this container locally, and provide a bash shell prompt.
* activate the virtual environment using `. /root/venv/bin/activate` .
* in order to save output of the training outside of the container use `-e /checkpoint/Results` as argument to coach.

## Usage under Windows
* download and unzip the repo, change directory to the `docker` folder using command line
* `docker build -f=Dockerfile.base -t=rlmot-base ..` to create the image
* will create a basic Coach installation along with the continuous control MOT environment.
* `docker run --rm --net host -v C:\checkpoint\:/checkpoint -it rlmot-base /bin/bash` will launch this container locally, and provide a bash shell prompt.
* activate the virtual environment using `. /root/venv/bin/activate` .
* in order to save output of the training outside of the container use `-e /checkpoint/Results` as argument to coach.

## other dockerfiles
* please refer to the [original README](README_coach.md) for more information on other Dockerfiles and images
