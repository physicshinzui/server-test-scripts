#!/bin/bash 
set -eu 

#singularity build --nv --fakeroot --sandbox gmx_gpu gmx_gpu.def
singularity build --nv --fakeroot gmx_gpu.sif gmx_gpu.def
