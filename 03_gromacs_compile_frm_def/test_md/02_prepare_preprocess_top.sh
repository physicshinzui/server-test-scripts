#!/bin/bash 
set -eu
. /etc/profile.d/modules.sh
module load cuda/11.2.146
module load python/3.8.3
module load gcc/8.3.0
module load cmake/3.21.3
module load openmpi intel-mpi/21.8.0
export CC=`which gcc`
export CXX=`which g++`
. ~/.bashrc

export PATH=/gs/hs1/hp230064/siida/software/gromacs-2022.5-plumed-2.8.3/build/bin:$PATH
export LD_LIBRARY_PATH=/gs/hs1/hp230064/siida/software/gromacs-2022.5-plumed-2.8.3/build/lib:$LD_LIBRARY_PATH

cat<<EOF
Q. Why this script must be executed? 

Answer: 
#include lines must not be in topol.top.
To remove them, the following command must be executed.
NOTE: 
    - This step is necessary to pass a topology file where 'ATOMNAME_' lines are written.
    - Modifying topol.top does not work due to the inclusion of #include lines.
      'plumed partial_tempering' can't understad the lines.
EOF

GMX=gmx_mpi
${GMX} grompp -f templates/em2.mdp \
              -c em1.gro \
              -p topol.top \
              -pp\
              -o em2.tpr #overwritten
