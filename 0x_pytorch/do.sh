#!/bin/bash 
set -eu 

# Create sif file
#singularity pull torch2.0.0-cuda11.70cudnn8.sif docker://pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

# Execute a training
cd test_MNIST_classification
## GPU
mkdir gpu; cd gpu
singularity exec --nvccli ../../torch2.0.0-cuda11.70cudnn8.sif python ../train.py | tee train.log
cd ..

## CPU
mkdir cpu; cd cpu
singularity exec ../../torch2.0.0-cuda11.70cudnn8.sif python ../train.py | tee train.log
cd ..
