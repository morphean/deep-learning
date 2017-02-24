import numpy as np
from accelerate.cuda import rand, cuda_compatible

def initWeights(noOfInputs, noOfHidden):
    return np.random.normal(0, noOfInputs ** -0.5, size=(noOfInputs, noOfHidden))

#input data
features = []


# network looks like
#        o              w11 w12
#       / \             w21 w22
#      h1  h2           w31 w32
#     / \  / \
#    x1  x2  x3

#Network size
N_input = 4
N_hidden = 3
N_output = 2

np.random.seed(42)

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1/(1+np.exp(-x))

shape = (N_input, N_hidden)

# Make some fake data
if cuda_compatible() is True:
    X = rand.normal(0, 1, (4))
else:
    X = np.random.randn(4)

weights_input_to_hidden = np.random.normal(0, scale=0.1, size=(N_input, N_hidden))
weights_hidden_to_output = np.random.normal(0, scale=0.1, size=(N_hidden, N_output))


# TODO: Make a forward pass through the network

hidden_layer_in = np.dot(X, weights_input_to_hidden)
hidden_layer_out = sigmoid(hidden_layer_in)

print('Hidden-layer Output:')
print(hidden_layer_out)

output_layer_in = np.dot(hidden_layer_out, weights_hidden_to_output)
output_layer_out = sigmoid(output_layer_in)

print('Output-layer Output:')
print(output_layer_out)

# Hidden-layer Output: (using consistent random seed
# [ 0.41492192  0.42604313  0.5002434 ]
# Output-layer Output:
# [ 0.49815196  0.48539772]