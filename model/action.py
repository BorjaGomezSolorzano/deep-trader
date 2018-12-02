from commons import constants
import tensorflow as tf
import os
import yaml

dirname = os.path.abspath(os.path.dirname(__file__))
filename = os.path.join(dirname, "../config/config.yaml")
config = yaml.load(open(filename, 'r'))

def get_action(input_standard, action, Ws, bs):
    l = len(config['n_layers'])

    input = tf.concat([input_standard, tf.reshape(action, (1, 1))], 0)
    hidden_layer_i = constants.f(tf.add(tf.matmul(input, Ws[0]), bs[0]))

    for i in range(1, l):
        hidden_layer_i = constants.f(tf.add(tf.matmul(hidden_layer_i, Ws[i]), bs[i]))

    output_layer = constants.f(tf.add(tf.matmul(hidden_layer_i, Ws[l]), bs[l]))

    return tf.transpose(output_layer)[0][0]