#!/bin/bash

# This script to be used after getting into the docker image

export cuBB_SDK=$(pwd)

# mkdir build

# cd build

# cmake .. -DCMAKE_TOOLCHAIN_FILE=cuPHY/cmake/toolchains/native

# Compile the code

# make -j $(nproc --all)

export CUDA_VISIBLE_DEVICES=0 # $(nvidia-smi -L|grep 'MIG 3g\.'| sed -n 's/.*(UUID: \(.*\))/\1/p')

echo $CUDA_VISIBLE_DEVICES

export CUDA_DEVICE_MAX_CONNECTIONS=8

export CUDA_MPS_PIPE_DIRECTORY=/tmp/$CUDA_VISIBLE_DEVICES

mkdir -p $CUDA_MPS_PIPE_DIRECTORY

export CUDA_MPS_LOG_DIRECTORY=/var

# Stop existing MPS

echo "Stop existing mps"

sudo -E echo quit | sudo -E nvidia-cuda-mps-control

# Start MPS

echo "Start mps"

sudo -E nvidia-cuda-mps-control -d

sudo -E echo start_server -uid 0 | sudo -E nvidia-cuda-mps-control
#sudo -E /opt/nvidia/cuBB/build/cuPHY-CP/cuphycontroller/examples/cuphycontroller_scf "P5G_FXN_GH" 
sudo -E /opt/nvidia/cuBB/build/cuPHY-CP/cuphycontroller/examples/cuphycontroller_scf "dyncore"

# exit 0
