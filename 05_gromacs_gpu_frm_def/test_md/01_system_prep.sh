#!/bin/bash 
set -eu 

GMX=gmx

fin=$1
${GMX} pdb2gmx -f $fin -o processed.gro -water tip3p -ignh

echo Protein | ${GMX} editconf -f processed.gro \
                -o newbox.gro    \
                -d 1.0           \
                -princ           \
                -bt cubic

${GMX} solvate -cp newbox.gro \
               -cs spc216.gro \
               -o  solv.gro   \
               -p  topol.top

${GMX} grompp -f templates/ions.mdp \
              -c solv.gro  \
              -p topol.top \
              -po mdout_ion.mdp \
              -o ions.tpr

echo "SOL" | ${GMX} genion \
             -s ions.tpr \
             -o solv_ions.gro \
             -p topol.top \
             -pname NA -nname CL \
             -neutral 
             #-conc 0.1 -neutral 

options="-ntmpi 1 -nt 16"
echo "Energy minimisation 1 ..."
${GMX} grompp -f templates/em1.mdp \
              -c solv_ions.gro \
              -r solv_ions.gro \
              -p topol.top \
              -po mdout_em1.mdp \
              -o em1.tpr -maxwarn 1
${GMX} mdrun -deffnm em1 $options

echo "Energy minimisation 2 ..."
${GMX} grompp -f templates/em2.mdp \
              -c em1.gro \
              -p topol.top \
              -po mdout_em2.mdp \
              -o em2.tpr -maxwarn 1
${GMX} mdrun -deffnm em2 $options


echo "Equilibriation at NPT"
${GMX} grompp -maxwarn 1 -o npt_eq.tpr -f inputs/npt_eq.mdp -p topol.top -c em2.gro -po mdout_npt_eq.mdp
${GMX} mdrun -deffnm npt_eq $options -pin on
