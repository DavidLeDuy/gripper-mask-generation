#!/bin/bash

# get script path
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
# exit if fails
set -e

cd $SCRIPTPATH"/project" 

#create data folders
cF(){
    if [ ! -d $1 ]
    then
        mkdir $1
        echo created folder $1
    else
        echo folder $1 already exists
    fi
}
cF data
cd data
cF train
cF test
cF val

echo script successfully run

# check if  virtualenv exists if not installing
cd $SCRIPTPATH
if [ -x "$(command -v virtualenv)" ]
    then
        virtualenv --python=python3.6 venv
    else
        echo "Installing virtualenv"
        pip install virtualenv
fi

# install req in venv
. venv/bin/activate
pip install -r requirements.txt

