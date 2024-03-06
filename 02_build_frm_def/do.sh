#!/bin/bash 
set -eu 

singularity build --fakeroot lolcow.sif lolcow.def
