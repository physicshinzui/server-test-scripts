#!/bin/bash 
set -eu 
log=$1
grep 'Mean loss' $log  | awk '{print $5}' > loss.dat 
grep 'Mean vali' $log | awk '{print $9}' > val_loss.dat 

gnuplot <<EOF
set terminal pdf
set output 'training_curve.pdf'
plot [][0:] 'loss.dat' w l title 'Training', 'val_loss.dat' w l title 'Validation'
EOF
