import os

from commons import constants
import tensorflow as tf
import yaml

dirname = os.path.abspath(os.path.dirname(__file__))
filename = os.path.join(dirname, "../config/config.yaml")
config = yaml.load(open(filename, 'r'))

def action(self, input_standard, action, Ws, bs):
    l = len(config['n_layers'])
    layer = input_standard
    for i in range(1, l):
        layer = constants.f(tf.add(tf.matmul(layer, Ws[i - 1]), bs[i - 1]))

    layer_augmented = tf.concat((layer, action), axis=1)
    a = constants.f(tf.add(tf.matmul(layer_augmented, Ws[l - 1]), bs[l - 1]))

    return a