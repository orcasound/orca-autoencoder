# Notes on docker usage to get 'known' python libraries and gpu access

# python version must be 3.6 or 3.7 for Ketos but maybe 3.8 for database_interface.py
	sudo add-apt-repository ppa:deadsnakes/ppa
	sudo apt-get update
	sudo apt-get install python3.8

# create directory for docker items  here docker_tf
# put requirements.txt and dockerfile in this directory
# cd to this directory and execute docker build

	docker build --no-cache -t ae/tensorflow-gpu .    # --no-cache will force requirements.txt dependencies to be rebuilt

# report:
	Successfully built f7c1f435be52
	Successfully tagged ae/tensorflow-gpu:latest



#  Map code and data volumes from local directories to /tmp/code and tmp/data directories inside the docker container:
#	Run the docker container

docker run -u $(id -u):$(id -g) -v /home/val/pythonFiles/DEC_nnFiles:/tmp/code:rw -v "/media/val/Seagate Portable Drive/WAVs_09_12_2021":/tmp/data:rw --gpus all -it ae/tensorflow-gpu bash

cd /tmp/data   # for data outputs
cd /tmp/code   #for python programs

# run python in the code directory

	python theNNpythonfile.py
	

# some randomn dock3r commands

docker container ls
docker container ls -a
docker images

docker rmi -f `docker images -q`  # get rid of everything so you can start over

docker stop 7887359a5a74   # docker stop CONTAINER ID

docker image rm -f 81ae0ebdb0b1

