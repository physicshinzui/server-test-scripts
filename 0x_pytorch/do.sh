#!/bin/bash 
set -eu 

#singularity build torch-24.02-py3-igpu.sif docker://nvcr.io/nvidia/pytorch:24.02-py3-igpu
#singularity build torch.sif docker://nvcr.io/nvidia/pytorch:24.02-py3
#singularity build torch.sif docker://nvcr.io/nvidia/pytorch:23.11-py3-igpu
singularity pull torch2.0.0-cuda11.70cudnn8.sif docker://pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

#singularity build --fakeroot torch.sif torch.def
