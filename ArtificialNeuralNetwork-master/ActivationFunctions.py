# Theo Satloff and Walker Griggs
# Artificial Neural Network Activation Functions
# May 2016

import numpy as np
np.seterr(all = 'ignore')

# Activation Functions
def sigmoid(x): # sigmoid(x) = 1/(1+e^-x)
    return 1 / (1 + np.exp(-x))

def dsigmoid(y): # sigmoid'(x) = sigmoid(x)(1-sigmoid(x))
    return y * (1.0 - y)

def tanh(x): # tanh = (e^x-x^-x)/(e^x+e^-x)
    return np.tanh(x)

def dtanh(y): # tanh' = 1 - tanh
    return 1 - y*y

# Sigmoid Approximations (Removes Exponents, Generally Faster)
# http://www.dontveter.com/bpr/art3.html#elliott1

def elliot(x):
    return 0.5*(x)/(1 + np.abs(x))+0.5

def softmax(w):
    e = np.exp(w - np.amax(w))
    dist = e / np.sum(e)
    return dist
