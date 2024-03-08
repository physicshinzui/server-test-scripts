#!/bin/bash 
set -eu 

singularity build --nv --fakeroot gmx_gpu.sif gmx_gpu.def
