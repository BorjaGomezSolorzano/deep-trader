from model import *

def action(input_standard, action, Ws, bs):
    l = len(n_layers)
    layer = input_standard
    for i in range(1, l):
        layer = constants.f(tf.add(tf.matmul(layer, Ws[i - 1]), bs[i - 1]))

    layer_augmented = tf.concat((layer, action), axis=1)
    a = constants.f(tf.add(tf.matmul(layer_augmented, Ws[l - 1]), bs[l - 1]))

    return a