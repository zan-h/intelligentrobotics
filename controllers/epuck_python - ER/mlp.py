#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Multi-layer perceptron
# Copyright (C) 2011  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import numpy as np

def sigmoid(x):
    ''' Sigmoid like function using tanh '''
    return np.tanh(x)

class MLP:
    ''' Multi-layer perceptron class. '''

    def __init__(self, *args):
        ''' Initialization of the perceptron with given sizes.  '''
        # Set random seed for weight initialization
        np.random.seed(42)

        self.shape = args
        n = len(self.shape[0])

        # Build layers
        self.layers = []
        # Input layer (+1 unit for bias)
        
        print("Chosen MLP Architecture:")
        print(self.shape[0])
        
        # Initialize input layer with bias node
        self.layers.append(np.ones(self.shape[0][0]+1))
        # Hidden layer(s) + output layer
        for i in range(1,n):
            self.layers.append(np.ones(self.shape[0][i]))

        # Build weights matrix (initialize with random values between -0.5 and 0.5)
        self.weights = []
        for i in range(n-1):
            self.weights.append(np.random.uniform(-0.5, 0.5, 
                                               (self.layers[i].size,
                                                self.layers[i+1].size)))
            
    def propagate_forward(self, data):
        ''' Propagate data from input layer to output layer. '''

        # Set input layer values, keeping bias as 1
        self.layers[0][0:-1] = data

        # Propagate from layer 0 to layer n-1 using sigmoid as activation function
        for i in range(1,len(self.shape[0])):
            # Calculate weighted sum and apply activation function
            self.layers[i][...] = sigmoid(np.dot(self.layers[i-1],self.weights[i-1]))

        # Return output layer activations
        return self.layers[-1]