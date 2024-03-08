#!/bin/bash 
set -eu 

# cpu gromacs
#singularity build --fakeroot --sandbox 01_gmx_cpu 01_gmx_cpu.def 
singularity build --fakeroot gmx_cpu.sif gmx_cpu.def 

# 02 gromacs on gpu
#singularity build --nv --fakeroot 02_gmx_gpu.sif 02_gmx_gpu.def
