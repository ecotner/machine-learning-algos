# TensorFlow implementation of a ResNet architecture with Inception blocks

This is still a work in progress.

## Goals:
- Be able to construct a convolutional neural network using residual skip connections and inception blocks in order to classify images from the CIFAR-10 dataset.
- Make efficient use of GPU resources so that transferring data between the CPU/GPU does not become a bottleneck.
  - Save the entire training set in GPU memory for quick retrieval.
- Make architecture flexible so that it can be adapted for object detection and neural style transfer projects.

## Done:
- Build ResNet class for making quick, customizable residual networks with Inception blocks
- Make pipeline for logging training/validation data from training
- Figure out how to save data sets on the GPU
- Calculate/log prediction accuracy as a metric
- Add weights and biases to variable collections for ease of applying regularization after constructing graph
- Add L1 and L2 regularization to training network
- Add in automatic noise injection to inputs
- Add in automatic dataset augmentation (random crops, flips, etc)
- Add dropout to network

## To Do:

- Add in batch normalization
- Add in automatic noise injection to weights
- Write script to run prediction on pictures using trained network
- Figure out how to do neural style transfer on the trained network
