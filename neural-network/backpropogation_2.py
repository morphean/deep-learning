import numpy as np
# import accelerate.cuda as cuda

from data_prep import features, targets, features_test, targets_test

# useGPU = cuda.cuda_compatible()

np.random.seed(21)

def sigmoid(x):
    # calculate sigmoid
    return 1 / (1+np.exp(-x))

#Hyper parameters
n_hidden = 2 # number of hidden units
epocs = 900
learn_rate = 0.042

n_records, n_features = features.shape

last_loss = None

#Initialise weights

weights_input_hidden = np.random.normal(scale=1 / n_features ** .5,
                                        size=(n_features, n_hidden))

weights_hidden_output = np.random.normal(scale=1 / n_features ** .5,
                                         size=n_hidden)

for e in range(epocs):
    delta_w_input_hidden = np.zeros(weights_input_hidden.shape)
    delta_w_hidden_output = np.zeros(weights_hidden_output.shape)

    for x, y in zip(features.values, targets):
        ## Forward pass
        # calcluate the output
        hidden_input = np.dot(x, weights_input_hidden)
        hidden_output = sigmoid(hidden_input)

        output = sigmoid(np.dot(hidden_output, weights_hidden_output))
        #backward pass
        # calculate the networks prediction error
        error = y - output

        # calculate error term for the output unit
        output_error_term = error * output * (1-output)
        ## pass thtough to hidden layer

        # caculate the hidden layer's contribution to the error
        hidden_error = np.dot(output_error_term, weights_hidden_output)

        # calculate the error for hidden layer
        hidden_error_term = hidden_error * hidden_output * (1-hidden_output)

        # update the change in weights
        delta_w_hidden_output += output_error_term * hidden_output
        delta_w_input_hidden += hidden_error_term * x[:, None]

    weights_hidden_output += learn_rate * delta_w_hidden_output / n_records
    weights_input_hidden += learn_rate * delta_w_input_hidden / n_records

    #out put the mean square on training set
    if e % (epocs / 10) == 0:
        hidden_output = sigmoid(np.dot(x, weights_input_hidden))
        out = sigmoid(np.dot(hidden_output, weights_hidden_output))
        loss = np.mean((out-targets)**2)

        if last_loss and last_loss < loss:
            print("WARN: Loss Increasing. Loss: ", loss)
        else:
            print("INFO: Loss: ", loss)


#Calculte accurayc on the test data
hidden = sigmoid(np.dot(features_test, weights_input_hidden))
out = sigmoid(np.dot(hidden, weights_hidden_output))
predictions = out >0.5
accuracy = np.mean(predictions==targets_test)
print ("Prediction accuracy: {:.3f}", format(accuracy));

