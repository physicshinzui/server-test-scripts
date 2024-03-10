#!/bin/bash 
set -eu 

singularity build --fakeroot gmx_cpu.sif gmx_cpu.def
