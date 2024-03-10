#!/bin/bash 
set -eu 

#singularity build --nv --fakeroot --sandbox gmx_gpu gmx_gpu.def

# Main
singularity build --nv --fakeroot gmx_gpu_hostcuda.sif gmx_gpu.def
singularity build --nvccli --fakeroot gmx_gpu_contcuda.sif gmx_gpu.def
