#!/bin/bash 
set -eu 
cat <<EOF 

A short MD run is performed for alanine dipeptide. 

EOF
cd test_md 
singularity exec ../gmx_gpu.sif ./01_system_prep.sh test_systems/02_pdbs/diala.pdb
#singularity exec --bind ../test_md:/home/siida ../gmx_gpu.sif ./01_system_prep.sh test_systems/02_pdbs/diala.pdb

