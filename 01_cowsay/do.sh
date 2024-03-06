#!/bin/bash 
set -eu
singularity pull library://lolcow
singularity exec lolcow_latest.sif cowsay Hi
