#!/bin/bash 
set -eu 

#singularity pull docker://nvidia/cuda:12.3.2-devel-ubi8
singularity exec --nv cuda_12.3.2-devel-ubi8.sif nvidia-smi > nvidia_smi.host 
singularity exec --nvccli cuda_12.3.2-devel-ubi8.sif nvidia-smi > nvidia_smi.container

