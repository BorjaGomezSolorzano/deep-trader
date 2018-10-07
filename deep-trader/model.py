# -*- coding: utf-8 -*-
"""
Created on 11/09/2018

@author: Borja
"""

import tensorflow as tf
from rlFunctions import Functions
from graphics.plot_results import Plot
import constants
import os
import yaml
import numpy as np

dirname = os.path.abspath(os.path.dirname(__file__))
filename = os.path.join(dirname, "../config/config.yaml")

class Model():

    def __init__(self):
        config = yaml.load(open(filename, 'r'))
        self.n_layers = config['n_layers']
        self.n_features = len(config['features_idx'])
        self.learning_rate = config['learning_rate']
        self.c = config['c']
        self.epochs = config['epochs']


    def train(self, i_train, o_train, i_test, o_test):

        series_test_length = i_test.shape[0]
        series_train_length = i_train.shape[0]
        len = series_train_length - self.n_layers[0]

        Ws, bs = self.optimize_all(len)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            init.run()
            for ep in range(self.epochs):
                feed_dict = {self.action_ph: np.zeros((1, 1))}
                for i in range(self.n_layers[0], series_train_length):
                    feed_dict[self.input_ph[i - self.n_layers[0]]] = i_train[(i - self.n_layers[0]):i, ]
                    aux = np.zeros((1, 1), dtype=constants.float_type_np)
                    aux[0][0] = o_train[i][0]
                    feed_dict[self.output_ph[i - self.n_layers[0]]] = aux

                reward, _ = sess.run([self.u, self.optimizer], feed_dict=feed_dict)

            Ws_real, bs_real = sess.run([Ws, bs])

            i_test_ph = tf.placeholder(tf.float32, shape=[self.n_layers[0], self.n_features])
            f_a_ph = tf.placeholder(tf.float32, [1, 1])
            c_a = f_a_ph

            action = self.get_action(i_test_ph, c_a, Ws_real, bs_real)

            functions = Functions()

            a = np.zeros((1, 1), dtype=constants.float_type_np)
            accum_rew = 0
            past_a = a
            actions = []
            rewards = []
            for i in range(self.n_layers[0], series_test_length):
                feed_dict = {}
                feed_dict[i_test_ph] = i_test[(i - self.n_layers[0]):i]
                feed_dict[f_a_ph] = a

                a_pred = sess.run(action, feed_dict=feed_dict)

                action_appended = a_pred.transpose()[0][0]
                actions.append(action_appended)

                a = np.zeros((1, 1), dtype=constants.float_type_np)
                a[0][0] = action_appended

                rew = functions.reward_array(o_test[i][0], self.c, past_a, a)
                accum_rew += rew[0][0]
                past_a = a

                rewards.append(accum_rew)

            plot = Plot()
            plot.plot_results(o_test, rewards, actions)


    def get_action(self, input, past_action, Ws, bs):
        """

        :param input:
        :param action:
        :param Ws:
        :param bs:
        :return:
        """

        # The standard memory
        standard_memory = tf.matmul(input, Ws[0][0:self.n_features])
        # The memory from the past
        recurrent_memory = tf.matmul(past_action ,tf.reshape(Ws[0][self.n_features], [1, self.n_layers[0]]))
        input_layer = constants.f(tf.add(tf.add(standard_memory, recurrent_memory), bs[0]))
        hidden_layer = constants.f(tf.add(tf.matmul(input_layer, Ws[1]), bs[1]))
        output_layer = constants.f(tf.add(tf.matmul(hidden_layer, Ws[2]), bs[2]))

        return output_layer

    def init_placeholders(self, len):
        self.input_ph = []
        self.output_ph = []
        self.action_ph = tf.placeholder(constants.float_type_tf, shape=[1, 1])
        for i in range(len):
            self.input_ph.append(tf.placeholder(constants.float_type_tf, shape=[self.n_layers[0], self.n_features]))
            self.output_ph.append(tf.placeholder(constants.float_type_tf, shape=[1, 1]))

    def init_weights_and_biases(self):

        Ws = [tf.Variable(tf.random_uniform([self.n_features + 1, self.n_layers[0]]), dtype=constants.float_type_tf), #Input
                   tf.Variable(tf.random_uniform([self.n_layers[0], self.n_layers[1]]), dtype=constants.float_type_tf), #Hidden
                   tf.Variable(tf.random_uniform([self.n_layers[1], 1]), dtype=constants.float_type_tf)] #Output

        bs = [tf.Variable(tf.zeros([self.n_layers[0]]), dtype=constants.float_type_tf), #Input
                   tf.Variable(tf.zeros([self.n_layers[1]]), dtype=constants.float_type_tf), #Hidden
                   tf.Variable(tf.zeros([1]), dtype=constants.float_type_tf)] #Output

        return Ws, bs

    def optimize_all(self, l):
        """

        :return:
        """

        self.init_placeholders(l)
        Ws, bs = self.init_weights_and_biases()

        functions = Functions()

        rewards = []
        self.action = self.action_ph
        for i in range(l):
            n_a = self.get_action(self.input_ph[i], self.action, Ws, bs)
            rewards.append(functions.reward(self.output_ph[i], self.c, n_a, 0 if i == 0 else self.action))
            self.action = n_a

        self.u = functions.utility(rewards)
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(-self.u)

        return Ws, bs
