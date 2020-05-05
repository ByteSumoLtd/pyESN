# Echo State Networks on the HyperSphere, in Python

[Echo State Networks](http://www.scholarpedia.org/article/Echo_state_network) are easy-to-train recurrent neural networks, a variant of [Reservoir Computing](https://en.wikipedia.org/wiki/Reservoir_computing). In some sense, these networks show how far you can get with nothing but a good weight initialisation.

This ESN implementation is relatively simple and self-contained, though it offers tricks like noise injection and teacher forcing (feedback connections), plus a zoo of dubious little hyperparameters.


# HyperSphere Implementation

This library is a fork of the original pyESN, that implements the Nature paper: [Echo State Networks with Self-Normalizing Activations on the Hyper-Sphere](https://arxiv.org/abs/1903.11691)

# Examples

An example of calling and training the ESN on the HyperSphere: 


