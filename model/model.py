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
        self.window_size = config['window_size']
        self.n_actions = config['n_actions']
        self.functions = Functions()

    def train(self, X, y, dates, instrument):

        if self.window_size < self.n_layers[0]: #FIXME Sacar esto fuera
            print('ERROR, NOT ENOUGH DATA')

        Ws, bs = self.weights_and_biases()

        input_ph, output_ph = self.init_placeholders()

        rewards = []
        actions = []
        action_ph = tf.placeholder(constants.float_type_tf, shape=())
        past_action = action_ph
        for i in range(self.window_size):
            action = self.get_action(input_ph[i], past_action, Ws, bs)
            rewards.append(self.functions.reward(output_ph[i], self.c, action, past_action))
            actions.append(action)
            past_action = action

        u = self.functions.utility(rewards)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(-u)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            accum_rew, a, past_a = 0, 0, 0
            actions,rewards, dates_o, instrument_o = [], [], [], []
            for j in range(self.n_layers[0], self.n_layers[0] + self.n_actions): #Execute n actions
                init.run()
                #Train
                for ep in range(self.epochs):
                    feed_dict = {action_ph: 0}
                    for index in range(self.window_size):
                        i = index + j
                        feed_dict[input_ph[index - self.n_layers[0]]] = X[(i - self.n_layers[0]):i, ].reshape((self.n_layers[0]*self.n_features,1))
                        feed_dict[output_ph[index - self.n_layers[0]]] = y[i]

                    acum_reward, _ = sess.run([u, optimizer], feed_dict=feed_dict)
                    #print("epoch: ", str(ep), ", accumulate reward: ", str(acum_reward))

                Ws_real, bs_real = sess.run([Ws, bs])

                i+=1

                dates_o.append(dates[i])
                instrument_o.append(instrument[i])

                #Test
                i_test_ph = tf.placeholder(constants.float_type_tf, shape=[self.n_layers[0] * self.n_features, 1])
                f_a_ph = tf.placeholder(constants.float_type_tf, shape=())

                action = self.get_action(i_test_ph, f_a_ph, Ws_real, bs_real)

                a = sess.run(action, feed_dict={i_test_ph:X[(i - self.n_layers[0]):i].reshape((self.n_layers[0] * self.n_features, 1)), f_a_ph:a})
                actions.append(a)

                rew = self.functions.reward(y[i], self.c, a, past_a)
                past_a = a

                accum_rew += rew
                rewards.append(accum_rew)

                print('action predicted: ', str(a), ', reward: ', str(rew), ', accumulated reward: ', str(accum_rew))

        return rewards, actions, dates_o, instrument_o

    def weights_and_biases(self):
        Ws = []
        bs = []
        dim_before = self.n_layers[0] * self.n_features + 1
        Ws.append(tf.Variable(tf.random_uniform([1, dim_before]), dtype=constants.float_type_tf))
        bs.append(tf.Variable(tf.zeros([dim_before]), dtype=constants.float_type_tf))

        for i in range(1, len(self.n_layers)):
            dim = self.n_layers[i]
            Ws.append(tf.Variable(tf.random_uniform([dim_before, dim]),dtype=constants.float_type_tf))
            bs.append(tf.Variable(tf.zeros([dim]), dtype=constants.float_type_tf))
            dim_before = dim

        bs.append(tf.Variable(tf.zeros([1]), dtype=constants.float_type_tf))
        Ws.append(tf.Variable(tf.random_uniform([dim_before, 1]), dtype=constants.float_type_tf))

        return Ws, bs

    def get_action(self, input_standard, action, Ws, bs):
        l = len(self.n_layers)

        input = tf.concat([input_standard, tf.reshape(action, (1, 1))], 0)
        hidden_layer_i = constants.f(tf.add(tf.matmul(input, Ws[0]), bs[0]))

        for i in range(1, l):
            hidden_layer_i = constants.f(tf.add(tf.matmul(hidden_layer_i, Ws[i]), bs[i]))

        output_layer = constants.f(tf.add(tf.matmul(hidden_layer_i, Ws[l]), bs[l]))

        return tf.transpose(output_layer)[0][0]

    def init_placeholders(self):
        input_ph = []
        output_ph = []
        for i in range(self.window_size):
            input_ph.append(tf.placeholder(constants.float_type_tf, shape=[self.n_layers[0] * self.n_features, 1]))
            output_ph.append(tf.placeholder(constants.float_type_tf, shape=()))

        return input_ph, output_ph