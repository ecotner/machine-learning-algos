# TensorFlow implementation of a ResNet architecture with Inception blocks

This is still a work in progress.

## Goals:
- Be able to construct a convolutional neural network using residual skip connections and inception blocks in order to classify images from the CIFAR-10 dataset.
- Make efficient use of GPU resources so that transferring data between the CPU/GPU does not become a bottleneck.
  - Prefetch batches and save them on the GPU memory.
  - Or maybe save the entire training set in GPU memory.
- Make architecture flexible so that it can be adapted for object detection and neural style transfer projects.