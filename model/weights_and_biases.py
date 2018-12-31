from model import *

def weights_and_biases():
    Ws = []
    bs = []

    l = len(n_layers)

    for i in range(1, l):
        dim = n_layers[i - 1] * n_features
        Ws.append(tf.Variable(tf.random_uniform([dim, dim], 0, 1)))
        bs.append(tf.Variable(tf.zeros([dim])))

    dim = n_layers[l - 1] * n_features + 1
    Ws.append(tf.Variable(tf.random_uniform([dim, 1], 0, 1)))
    bs.append(tf.Variable(tf.zeros([1])))

    return Ws, bs