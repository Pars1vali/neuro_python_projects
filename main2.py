import numpy as np
weight = [[0.1, 0.1, -0.3],
          [0.1, 0.2, 0.0],
          [0.0, 1.3, 0.1]]

def neural_network(input_data, weight_data):
    output = [0, 0, 0]
    for i in range(len(output)):
        output[i] = ele_mul(input_data[i], weight_data[i])
    return output

def ele_mul(vector, weight_current):
    output = 0
    for i in range(len(weight_current)):
        output +=vector * weight_current[i]
    return output

def get_w_delta(inp, whgt):
    output = np.zeros((len(inp), len(whgt)))
    for i in range(len(inp)):
        for j in range(len(whgt)):
            elem = whgt[i]
            output[i][j] = inp[i] * elem[j]
    return output

toes = [8.5, 9.5, 9.9, 9.0, 8.8, 9,3, 9.0, 8.8]
wlec = [0.65, 0.8, 0.8, 0.9, 0.9, 0.6, 0.8, 0.87]
nfans = [1.2, 1.3, 0.5, 1.0, 1.2, 1.3, 1.0, 0.9]

hurt = [0.1, 0.0, 0.0, 0.1, 0.1, 0.0, 0.1, 0.0]
win = [1, 1, 0, 1, 0, 1, 1, 1]
sad = [0.1, 0.0, 0.1, 0.2, 0.5, 0.1, 0.2, 0.1]

momemt = ["hurt", "win", "sad"]

alpha = 0.0
for i in range(8):
    input = [toes[i], wlec[i], nfans[i]]
    true = [hurt[i],win[i], sad[i]]

    pred = neural_network(input, weight)

    error = [0, 0, 0]
    delta = [0, 0, 0]

    for s in range(len(error)):
        error[s] = (pred[s]-true[s]) ** 2
        delta[s] = pred[s] - true[s]

    weight_delta = get_w_delta(input, weight)
    for z in range(len(input)):
        for j in range(len(weight)):
            weight[z][j] -= alpha * weight_delta[z][j]

    print("Iteration " + str(i) + "\n------")
    for k in range(len(error)):
        print(  " Moment - " + str(momemt[k]) + ", True - " + str(true[k]) + " Prediction - " + str(pred[k]) + " Error - " + str(error[k]))
    print("-----")