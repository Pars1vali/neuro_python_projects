import numpy as np

np.random.seed(1)


def relu(x):  # this function return matrix where if before had negative numbers function change of them zero, it's
    # work as follows - x - element matrix, if element > 0 then return this number else return zero
    return (x > 0) * x  # its work  - if x> 0 we get true == 1, and we multiply 1 out x and get our positive x, but if
    # x < 0, we get false == 0, we multiply 0 out x, and get 0, because x is negative number


def relu2deriv(output): # if output > 0 we get true == 1 and return it, if < 0 return 0==false
    return output > 0


aplha = 0.2
hidden_size = 4  # size matrix

streetlights = np.array([[1, 0, 1],  # training example
                         [0, 1, 1],
                         [0, 0, 1],
                         [1, 1, 1]])

walk_vs_stop = np.array([[1, 1, 0, 0]]).T  # training example

weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1  # initialized the value weights between zero and first layout,
# matrix has 3 line and 4 column, because we have 3 input in zero layout and 4 intermediate in first layout
weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1  # initialized the value weights between first and second layout
# matrix has 4 line and 1 column, because we have 1 intermediate in first layout and 1 output in second layout

for iteration in range(60):  # 60 iter in correcting neuron network
    layer_2_error = 0  # error second layout
    for i in range(len(streetlights)):  # for everyone training example
        layer_0 = streetlights[i:i + 1]  # get training example and put he value in input
        layer_1 = relu(np.dot(layer_0, weights_0_1))  # count value neurons 1 layout - > for this we multiply matrix
        # input out matrix weight, and we get for neuron, farther we put this matrix in function relu
        layer_2 = np.dot(layer_1, weights_1_2)  # create layout output, we multiply matrix neuron first layout out
        # matrix weights, which connected layout 1 and their neurons and layout second with him neuron

        layer_2_error += np.sum((layer_2 - walk_vs_stop[i:i + 1]) ** 2)  # we count error layout 2, second layout,
        # layout output, for this we subtract from neuron layout 2, our prognoses - right value, which we need to get,
        # we get error - but we need only positive number and on it, we multiply our error by yourself and np.sum -
        # we get a number error nor matrix error, and we add error in the variable

        layer_2_delta = (walk_vs_stop[i:i + 1] - layer_2)  # we get clear error, now we have known what if we want
        # to get right result we need correct out this value layout 1, we get matrix[1, 1]
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1) # we multiply matrix

        weights_0_1 += aplha * layer_0.T.dot(layer_1_delta)

        weights_1_2 += aplha * layer_1.T.dot(layer_2_delta)
    #
    # if iteration % 10 == 9:
    #     print("Error: " + str(layer_2_error))

input = np.array([[0, 1, 0]])
layer_0_my = input
layer_1_my = relu(np.dot(layer_0_my, weights_0_1))
layer_2_my = np.dot(layer_1_my, weights_1_2)
if(np.sum(layer_2_my)>0.9):
    print("go")
else:print("stop")