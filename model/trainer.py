# -*- coding: utf-8 -*-
"""
Created on 11/09/2018

@author: Borja
"""

import os

from commons import constants
import tensorflow as tf
import yaml
from sklearn.preprocessing import MinMaxScaler
from model.rlFunctions import utility, reward_tf, reward_np
import numpy as np
from model.weights_and_biases import weights_and_biases
from model.action import get_action

dirname = os.path.abspath(os.path.dirname(__file__))
filename = os.path.join(dirname, "../config/config.yaml")
config = yaml.load(open(filename, 'r'))

class Model():

    def __init__(self):
        self.n_layers = config['n_layers']
        self.n_features = len(config['features_idx'])
        self.learning_rate = config['learning_rate']
        self.c = config['c']
        self.epochs = config['epochs']
        self.window_size = config['window_size']
        self.n_actions = config['n_actions']


    def execute(self, X, y, dates, instrument):

        tf.set_random_seed(1)

        Ws, Wa, bs = weights_and_biases()

        input_ph = []
        output_ph = []
        for i in range(self.window_size):
            input_ph.append(tf.placeholder(constants.float_type_tf, shape=[self.n_layers[0] * self.n_features, 1]))
            output_ph.append(tf.placeholder(constants.float_type_tf, shape=()))

        rewards_train = []
        action_train_ph = tf.placeholder(constants.float_type_tf, shape=())
        past_action_train = action_train_ph
        for i in range(self.window_size):
            action_train = get_action(input_ph[i], past_action_train, Ws, Wa, bs)
            rewards_train.append(reward_tf(output_ph[i], self.c, action_train, past_action_train))
            past_action_train = action_train

        u = utility(rewards_train)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(-u)

        self.scaler = MinMaxScaler(feature_range=(-1, 1))

        X_transformed = self.scaler.fit_transform(np.copy(X[0:(self.n_layers[0] + self.window_size + self.n_actions), ]))

        accum_rewards = 0
        a, past_a = 0, 0
        actions_returned, simple_rewards, dates_o, instrument_o, rew_epochs = [], [], [], [], [0 for k in range(self.epochs)]

        init = tf.initialize_all_variables()
        with tf.Session() as sess:

            for j in range(self.n_layers[0], self.n_layers[0] + self.n_actions):

                sess.run(init)

                #Train
                for ep in range(self.epochs):
                    feed_dict = {action_train_ph: 0}
                    for index in range(self.window_size):
                        i = index + j
                        x=X_transformed[((i+1) - self.n_layers[0]):(i+1)].reshape((self.n_layers[0]*self.n_features,1))
                        feed_dict[input_ph[index]] = x
                        feed_dict[output_ph[index]] = y[i]

                    acum_reward, _ = sess.run([u, optimizer], feed_dict=feed_dict)
                    rew_epochs[ep] += acum_reward
                    #print("epoch: ", str(ep), ", accumulate reward: ", str(acum_reward), ', ', aa)

                Ws_real, Wa_real, bs_real = sess.run([Ws, Wa, bs])

                i+=1

                dates_o.append(dates[i])
                instrument_o.append(instrument[i])

                #Test
                i_test_ph = tf.placeholder(constants.float_type_tf, shape=[self.n_layers[0] * self.n_features, 1])
                f_a_ph = tf.placeholder(constants.float_type_tf, shape=())

                a_test = get_action(i_test_ph, f_a_ph, Ws_real, Wa_real, bs_real)

                x = X_transformed[((i+1) - self.n_layers[0]):(i+1),].reshape((self.n_layers[0] * self.n_features, 1))
                a = sess.run(a_test, feed_dict={i_test_ph:x, f_a_ph:a})
                actions_returned.append(a)

                rew = reward_np(y[i], self.c, a, past_a)
                accum_rewards += rew
                past_a = a

                simple_rewards.append(rew)

                print(str(i), ' action predicted: ', str(a), ', reward: ', str(rew), ', accumulated reward: ', str(accum_rewards))

        for k in range(self.epochs):
            rew_epochs[k] /= self.epochs

        return simple_rewards, actions_returned, dates_o, instrument_o, rew_epochs