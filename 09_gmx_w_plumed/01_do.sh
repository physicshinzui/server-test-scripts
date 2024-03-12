#!/bin/bash 
set -eu 
singularity build --nv --fakeroot gmx_gpu_w_plumed.sif gmx_gpu_w_plumed.def
