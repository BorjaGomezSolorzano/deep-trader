# -*- coding: utf-8 -*-
"""
Created on 11/09/2018

@author: Borja
"""

import os

from commons import constants
import tensorflow as tf
import yaml

from model.rlFunctions import Functions

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
        self.functions = Functions()

    def train(self, i_train, o_train, i_test, o_test):

        series_test_length = i_test.shape[0]
        series_train_length = i_train.shape[0]
        len = series_train_length - self.n_layers[0]

        Ws, bs = self.init_weights_and_biases()

        u, optimizer, input_ph, output_ph, action_ph = self.optimize_all(Ws, bs, len)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            init.run()
            for ep in range(self.epochs):
                feed_dict = {action_ph: 0}
                for i in range(self.n_layers[0], series_train_length):
                    feed_dict[input_ph[i - self.n_layers[0]]] = i_train[(i - self.n_layers[0]):i, ].reshape((self.n_layers[0]*self.n_features,1))
                    feed_dict[output_ph[i - self.n_layers[0]]] = o_train[i]

                reward,  _ = sess.run([u, optimizer], feed_dict=feed_dict)
                print("epoch: " + str(ep) + ", reward: " + str(reward))

            Ws_real, bs_real = sess.run([Ws, bs])

            print('Weights')
            print(Ws_real)
            print('Biases')
            print(bs_real)

            i_test_ph = tf.placeholder(constants.float_type_tf, shape=[self.n_layers[0] * self.n_features, 1])
            f_a_ph = tf.placeholder(constants.float_type_tf, shape=())

            action = self.get_action(i_test_ph, f_a_ph, Ws_real, bs_real)

            a = 0
            accum_rew = 0
            past_a = a
            actions = []
            rewards = []
            for i in range(self.n_layers[0], series_test_length):

                a = sess.run(action, feed_dict={i_test_ph:i_test[(i - self.n_layers[0]):i].reshape((self.n_layers[0] * self.n_features, 1)),
                                                     f_a_ph:a})
                actions.append(a)

                print('action predicted: ' + str(a))

                rew = self.functions.reward(o_test[i], self.c, a, past_a)
                past_a = a

                accum_rew += rew
                rewards.append(accum_rew)

        return rewards, actions

    def init_weights_and_biases(self):
        l=len(self.n_layers)
        Ws = []
        bs = []
        dim_before = self.n_layers[0] * self.n_features + 1
        Ws.append(tf.Variable(tf.random_uniform([1, dim_before]), dtype=constants.float_type_tf)) #Input layer
        bs.append(tf.Variable(tf.zeros([dim_before]), dtype=constants.float_type_tf))

        for i in range(1, l): #Hidden layers
            dim = self.n_layers[i]
            Ws.append(tf.Variable(tf.random_uniform([dim_before, dim]),dtype=constants.float_type_tf))
            bs.append(tf.Variable(tf.zeros([dim]), dtype=constants.float_type_tf))
            dim_before = dim

        bs.append(tf.Variable(tf.zeros([1]), dtype=constants.float_type_tf))
        Ws.append(tf.Variable(tf.random_uniform([dim_before, 1]), dtype=constants.float_type_tf)) #Output layer

        return Ws, bs

    def get_action(self, input_standard, action, Ws, bs):
        l = len(self.n_layers)

        input = tf.concat([input_standard, tf.reshape(action, (1, 1))], 0)
        hidden_layer_i = constants.f(tf.add(tf.matmul(input, Ws[0]), bs[0])) #Input layer

        for i in range(1, l): #Hidden layers
            hidden_layer_i = constants.f(tf.add(tf.matmul(hidden_layer_i, Ws[i]), bs[i]))

        output_layer = constants.f(tf.add(tf.matmul(hidden_layer_i, Ws[l]), bs[l])) #Output layer

        return tf.transpose(output_layer)[0][0] #tf.sign(tf.transpose(output_layer)[0][0])

    def init_placeholders(self, len):
        input_ph = []
        output_ph = []
        for i in range(len):
            input_ph.append(tf.placeholder(constants.float_type_tf, shape=[self.n_layers[0] * self.n_features, 1]))
            output_ph.append(tf.placeholder(constants.float_type_tf, shape=()))

        return input_ph, output_ph

    def optimize_all(self, Ws, bs, l):

        input_ph, output_ph = self.init_placeholders(l)

        rewards = []
        action_ph = tf.placeholder(constants.float_type_tf, shape=())
        past_action = action_ph
        for i in range(l):
            action = self.get_action(input_ph[i], past_action, Ws, bs)
            rewards.append(self.functions.reward(output_ph[i], self.c, action, past_action))
            past_action = action

        u = self.functions.utility(rewards)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(-u)

        return u, optimizer, input_ph, output_ph, action_ph
