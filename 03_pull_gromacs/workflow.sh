/#!/bin/bash 


singuilarity run --bind 1536:/mnt gromacs_2018.2.sif gmx grompp 
