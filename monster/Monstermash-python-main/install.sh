#!/bin/bash
sudo --stdin apt-get update 
sudo --stdin apt-get upgrade -y 
sudo --stdin apt-get install build-essential -y cmake -y libsdl2-dev -y wget -y unzip git -y

if [ ! -d ./monster-mash/build ] ; then
    mkdir ./monster-mash/build
fi

if [ ! -d ./monster-mash/build/Release ] ; then
    mkdir ./monster-mash/build/Release
else
    rm -r ./monster-mash/build/Release
    mkdir ./monster-mash/build/Release
fi
cd ./monster-mash/build/Release
cmake -DCMAKE_BUILD_TYPE=Release ../../src && make
cp ./monstermash.so ../../../monstermash.so