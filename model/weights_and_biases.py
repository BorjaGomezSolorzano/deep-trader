from commons import constants
import tensorflow as tf
import yaml
import os

dirname = os.path.abspath(os.path.dirname(__file__))
filename = os.path.join(dirname, "../config/config.yaml")
config = yaml.load(open(filename, 'r'))

def weights_and_biases():
    Ws = []
    bs = []
    dim_before = config['n_layers'][0] * len(config['features_idx']) + 1
    Ws.append(tf.Variable(tf.random_uniform([1, dim_before]), dtype=constants.float_type_tf))
    bs.append(tf.Variable(tf.zeros([dim_before]), dtype=constants.float_type_tf))

    for i in range(1, len(config['n_layers'])):
        dim = config['n_layers'][i]
        Ws.append(tf.Variable(tf.random_uniform([dim_before, dim]) ,dtype=constants.float_type_tf))
        bs.append(tf.Variable(tf.zeros([dim]), dtype=constants.float_type_tf))
        dim_before = dim

    Ws.append(tf.Variable(tf.random_uniform([dim_before, 1]), dtype=constants.float_type_tf))
    bs.append(tf.Variable(tf.zeros([1]), dtype=constants.float_type_tf))

    return Ws, bs