# A cat versus car detection problem on visual images with multi-layer NN

The data file assign2_data1.h5 contains the variables trainims (training images) and testims (testing images) along with the ground truth labels trainlbls and testlbls. _**Stochastic gradient descent on mini-batches**_ is implemented.

Measure the error term as a function of epoch number, where an epoch is a single pass through all images in the training set. Two different error metrics will be calculated: the mean squared error and the mean classification error (percentage of correctly classfied images). Record the error metrics for each epoch separately for the training samples and the testing samples.

## Part A

Using the backpropagation algorithm, design a multi-layer neural network with a single hidden layer. Assume a hyperbolic tangent activation function for all neurons. Experiment with different number of neurons N in the hidden layer, initialization of weight and bias terms, and mini-batch sample sizes. Assuming a learning rate of η ∈ [0.1 0.5], select a particular set of parameters that work well. Using the selected parameters, run backpropagation until convergence. Plot the learning curves as a function of epoch number for training squared error, testing squared error, training classification error, and testing classsification error.

## Part B

Describe how the squared-error and classification error metrics evolve over epochs for the training versus the testing sets? Is squared error an adequate predictor of classification error?

## Part C

Train separate neural networks using substantially smaller and larger number of hidden-layer neurons (N_low and N_high). Plot the learning curves for all error metrics, overlaying the results for N_low, N_high and N prescribed in part a.

## Part D

Design and train a separate network with two hidden layers. Assuming a learning rate of η ∈ [0.1 0.5], select a particular set of parameters that work well. Plot the learning curves for all error metrics, and comparatively discuss the convergence behavior and classification performance of the two hidden-layer network with respect to the network in part a.

## Part E

Assuming a momentum coefficient of α ∈ [0.1 0.5], retrain the neural network described in part d. Select a particular set of parameters that work well. Plot the learning curves for all error metrics, and comparatively discuss the convergence behavior with respect to part d.
