
# Echo State Networks on the HyperSphere, in Python

[Echo State Networks](http://www.scholarpedia.org/article/Echo_state_network) are a type of NN that are not trained with back propagation. 

They are large randomly connected networks, initialised with random connections and weights. You feed them sequences, and the random weights, connections and feedback loops, excite the network deterministically - then after the sequence, you take what is known as a "readout" that represents the final state at the end of the sequence.

Using the readout, the output state is passed as features to train a simple model from which to predict the target value(s) sought. 

Like all NNs, there are tunable hyper-parameters and schemes you can alter with ESNs too.

To tune an echo state network, the first port of call is to adjust the network structure. Adjusting the size of the reservoir is the main parameter. Also, it can be done through altering a parameter similar to "drop out", called Sparsity, which zeros out a random selection of the fully connected edges between nodes. 

An additional method is to alter (or scale) the random weights used when initialising the network. This can be done in a coarse way by changing the random seed, and in a focussed way by adjusting a parameter called the the Spectral Radius. The objective of this tuning to force the the network into a state known as the Edge of Chaos. It is widely reported, however, that ESNs are highly sensitive to minor changes in Spectral Radius making them difficult to tune, which affects performance.  

While spectral radius is difficult to tune, another class of parameter can help, which tunes the activation functions - and a recent paper has proposed to project the activation values onto a hypersphere of radius r, which normalises them (and thus stopping them from exploding, or vanishing). This approach of self-normalisation also has the side effect of stabilizing the network, irrespective of spectral radius. 

To investigate this, I've implemented the hyper-sphere normalisation method from the paper. The paper can be found here: http://www.nature.com/articles/s41598-019-50158-4    

The promise of the new methods - is ESN tuning that is less prone to failure, and generally an opportunity to better tune models to deliver better accuracy.

The Mackey notebook has a worked example. Hopefully, the twp parameters, Spherical Radius + Spectral Radius, can together allow for systematic tuning of the networks in a principled way that can improve stability and performance.  

This ESN implementation is relatively simple and self-contained, though it offers tricks like noise injection and teacher forcing (feedback connections), plus a zoo of dubious little hyperparameters.


# HyperSphere Implementation

This library is a fork of the original pyESN, that implements the paper found on arxiv here: [Echo State Networks with Self-Normalizing Activations on the Hyper-Sphere](https://arxiv.org/abs/1903.11691)

# Examples

An example of calling and training the ESN on the HyperSphere: 

- [learning a Mackey-Glass system](http://nbviewer.ipython.org/github/cknd/pyESN/blob/master/mackey.ipynb)
An example of using DEAP to tune the ESN using genetic search:
- [Genetic Tuning of ESNs on the HyperSphere using DEAP](http://nbviewer.ipython.org/github/ByteSumoLtd/pyESN/blob/master/GeneticallyTuned-pyESN-withSphericalActivations.ipynb)
