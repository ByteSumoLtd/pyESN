# Echo State Networks on the HyperSphere, in Python

[Echo State Networks](http://www.scholarpedia.org/article/Echo_state_network) are easy-to-train recurrent neural networks, a variant of [Reservoir Computing](https://en.wikipedia.org/wiki/Reservoir_computing). 

The benefit of ESNs are they they are not tuned using back-progation, and as such, they offer many interesting opportunities.
Geoffrey Hinton [introduces Echo State Networks](https://www.youtube.com/watch?v=prXjoD9rEHo) in this video, which is worth watching.

### HyperSphere Activation Implementation

The ByteSumo implementation is a fork of the original pyESN code, that implements the Nature Scientific Reports paper: [Echo State Networks with Self-Normalizing Activations on the Hyper-Sphere](https://arxiv.org/abs/1903.11691)

(Many thanks to Pietro Verzelli for discussions and help in explaining things to me).

This ESN implementation has a new activation function that helps to stablise the ESN on the "Edge of Chaos" across the Spectral Radius range. It's a beta implementation, and it seems to work well. If you spot ways to improve my code, or create interesting example notebooks using this code, please offer a pull request and I'll update the repo.

### Examples

##### Learning Mackey Glass - a worked example

Examples of training ESNs, having HyperSphere Activations: 

[Predicting Mackey Glass - testing ESNs having HyperSphere Activations](https://github.com/ByteSumoLtd/pyESN/blob/master/mackey.ipynb)

##### Automatic Hyperparameter Tuning of ESNs  - Using DEAP to do genetic search for good parameters

The new hypersphere activation functions allow for practical hyperparameter searching to tune Echo State Networks, and the example code linked below illustrates how to do it using [DEAP](https://github.com/DEAP/deap). The example illustrates that using genetic search works very well!

[Genetic Tuning of an ESN on the HyperSphere, via DEAP](https://github.com/ByteSumoLtd/pyESN/blob/master/GeneticallyTuned-pyESN-withSphericalActivations.ipynb)

### Screenshot

![Mackey Glass prediction](https://github.com/ByteSumoLtd/pyESN/blob/master/Screenshot%202020-05-08%20at%2012.11.19.png)
