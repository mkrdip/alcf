#!/bin/sh
#COBALT -t 20
#COBALT -n 1
#COBALT -q pubnet-debug
#COBALT -A datascience
#COBALT --attrs nox11

IMG=/soft/datascience/singularity/tensorflow/centos7-cuda9.0-tf1.12.img

singularity exec --nv ${IMG} python cifar10-gpu.py --log_dir ./multi_gpu_mtf_cifar_model/
