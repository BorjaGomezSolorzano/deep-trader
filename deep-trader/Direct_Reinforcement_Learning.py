import numpy as np
import tensorflow as tf
from rlFunctions import Functions


class Model():

    def __init__(self, series_train_length, features_len, learning_rate, look_back, c, n_layers):
        self.series_train_length = series_train_length
        self.features_len=features_len
        self.look_back=look_back
        self.features_len=features_len
        self.learning_rate=learning_rate
        self.c=c
        self.n_layers = n_layers


    def get_test_model(self):
        input_test_ph = tf.placeholder(tf.float32, shape=[None, self.look_back, self.features_len])
        output_test_ph = tf.placeholder(tf.float32, shape=[None, 1])

        h = tf.layers.dense(units=self.n_layers[0], inputs=input_test_ph, activation=tf.nn.tanh)
        h = tf.layers.dense(units=self.n_layers[1], inputs=h, activation=tf.nn.tanh)
        h = tf.layers.dense(units=1, inputs=h, activation=tf.nn.tanh)

        return h, input_test_ph, output_test_ph

    def get_model(self):
        input_train_ph = []
        for _ in range(self.series_train_length):
            input_train_ph.append(tf.placeholder(tf.float32, shape=[None, self.look_back, self.features_len]))

        output_train_ph = []
        for _ in range(self.series_train_length):
            output_train_ph.append(tf.placeholder(tf.float32, shape=[None, 1]))

        functions = Functions()

        actions = []
        rewards = []
        action_out = np.zeros((1,1))
        for t in range(1, self.series_train_length):
            h = tf.layers.dense(units=10, inputs=input_train_ph[t],activation=tf.nn.tanh)
            h = tf.layers.dense(units=10, inputs=h, activation=tf.nn.tanh)
            h = tf.layers.dense(units=1, inputs=h, activation=tf.nn.tanh)
            actions.append(h)
            if t==0:
                rewards.append(tf.reduce_sum(functions.reward(output_train_ph[t], self.c, h, 0)))
            else:
                rewards.append(tf.reduce_sum(functions.reward(output_train_ph[t], self.c, h, action_out)))
            action_out = h

        u = functions.utility(rewards)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(-u)

        return optimizer, u, input_train_ph, output_train_ph
