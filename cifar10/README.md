# CIFAR-10

This folder contains code for CIFAR-10 model built using Mesh TensorFlow in model-parallel to compare 1 vs 2 GPU performance. 

To run it on Single GPU is `cifar10.py`, please notice it has only 2 Layers. So you need to add layers yourself or copy code from `cifar10-gpu.py` and paste it in this file.

To run it on multi GPU use `cifa10-gpu.py`. Change epochs, model directory and play with layers to measure the performance.
