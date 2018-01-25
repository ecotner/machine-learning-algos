# Architectures

A collection of useful scripts for building deep neural networks in short time.

- mlp: A simple multi-layer perceptron (MLP) capable of building simple fully-connected networks.
	- Need to add train() and predict() methods
	- Need to add way of outputting architecture
	- Add way of saving model
- ResNet: More complex script capable of building a convolutional neural network (CNN) using convolutional Inception blocks and residual skip connections. Supports a host of bells and whistles such as dropout and batch normalization.
	- Add train() and predict() methods
	- Fix the janky way of saving the computational graph