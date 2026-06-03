# Simple Neural Network using Backpropagation

import numpy as np

# Input Data
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Expected Output
Y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of Sigmoid
def derivative(x):
    return x * (1 - x)

# Random weights
np.random.seed(1)

# Input to Hidden Layer weights
w1 = np.random.rand(2, 2)

# Hidden to Output Layer weights
w2 = np.random.rand(2, 1)

# Training the Network
for i in range(10000):

    # Forward Propagation

    hidden_layer = sigmoid(np.dot(X, w1))

    output_layer = sigmoid(np.dot(hidden_layer, w2))

    # Find Error

    error = Y - output_layer

    # Backpropagation

    d_output = error * derivative(output_layer)

    hidden_error = d_output.dot(w2.T)

    d_hidden = hidden_error * derivative(hidden_layer)

    # Update Weights

    w2 += hidden_layer.T.dot(d_output)

    w1 += X.T.dot(d_hidden)

# Final Output
print("Output after Training:\n")

print(output_layer)
