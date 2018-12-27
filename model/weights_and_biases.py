import os

import tensorflow as tf
import yaml

dirname = os.path.abspath(os.path.dirname(__file__))
filename = os.path.join(dirname, "../config/config.yaml")
config = yaml.load(open(filename, 'r'))

def weights_and_biases():
    Ws = []
    bs = []

    ls = config['n_layers']
    l = len(config['n_layers'])

    for i in range(1, l):
        dim = ls[i - 1] * self.n_features
        Ws.append(tf.Variable(tf.random_uniform([dim, dim], 0, 1)))
        bs.append(tf.Variable(tf.zeros([dim])))

    Ws.append(tf.Variable(tf.random_uniform([ls[l - 1] * self.n_features + 1, 1], 0, 1)))
    bs.append(tf.Variable(tf.zeros([1])))

    return Ws, bs