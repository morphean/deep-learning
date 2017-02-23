import numpy as np

def initWeights(noOfInputs, noOfHidden):
    return np.random.normal(0, noOfInputs ** -0.5, size=(noOfInputs, noOfHidden))

#input data
features = []

# number of records and input units
n_records, n_inputs = features.shape

# number of hidden units
n_hidden = 2

# network looks like
#        o              w11 w12
#       / \             w21 w22
#      h1  h2           w31 w32
#     / \  / \
#    x1  x2  x3
#

weights_input_to_hidden = initWeights(n_inputs, n_hidden)
