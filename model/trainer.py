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

dirname = os.path.abspath(os.path.dirname(__file__))
filename = os.path.join(dirname, "../config/config.yaml")
config = yaml.load(open(filename, 'r'))

class Model():

    def __init__(self):
        self.n_features = len(config['features_idx'])

    def flat(self, X_transformed, i):
        x1 = X_transformed[((i + 1) - config['n_layers'][0]):(i + 1)]
        l1 = config['n_layers'][0]
        l = l1 * self.n_features
        x = np.zeros((1, l))
        for k in range(self.n_features):
            for j in range(l1):
                x[0][k * l1 + j]=x1[j][k]

        return x


    def execute(self, X, y, dates, instrument):

        tf.set_random_seed(1)

        input_ph = []
        output_ph = []
        for i in range(config['window_size']):
            input_ph.append(tf.placeholder(constants.float_type_tf, shape=[1, config['n_layers'][0] * self.n_features]))
            output_ph.append(tf.placeholder(constants.float_type_tf, shape=()))

        Ws, bs = self.weights_and_biases()

        rewards_train = []
        action_train_ph = tf.placeholder(constants.float_type_tf, shape=(1,1))
        past_action_train = action_train_ph
        for i in range(config['window_size']):
            action_train = self.action(input_ph[i], past_action_train, Ws, bs)
            rewards_train.append(reward_tf(output_ph[i], action_train[0][0], past_action_train[0][0]))
            past_action_train = action_train

        u = utility(rewards_train)
        optimizer = tf.train.AdamOptimizer(learning_rate=config['learning_rate']).minimize(-u)

        self.scaler = MinMaxScaler(feature_range=(-1, 1))

        X_transformed = self.scaler.fit_transform(np.copy(X[0:(config['n_layers'][0] + config['window_size'] + config['n_actions'])]))

        accum_rewards = 0
        past_a = np.zeros((1,1))
        actions_returned, simple_rewards, dates_o, instrument_o, rew_epochs = [], [], [], [], [0 for k in range(config['epochs'])]

        init = tf.initialize_all_variables()
        with tf.Session() as sess:

            for j in range(config['n_layers'][0], config['n_layers'][0] + config['n_actions']):

                sess.run(init)

                #Train
                for ep in range(config['epochs']):
                    feed_dict = {action_train_ph: np.zeros((1,1))}
                    for index in range(config['window_size']):
                        i = index + j
                        x = self.flat(X_transformed, i)
                        feed_dict[input_ph[index]] = x
                        feed_dict[output_ph[index]] = y[i]

                    acum_reward, _, a, rr = sess.run([u, optimizer, action_train, rewards_train], feed_dict=feed_dict)
                    rew_epochs[ep] += acum_reward

                i+=1

                # Test

                dates_o.append(dates[i])
                instrument_o.append(instrument[i])

                actions_returned.append(a[0][0])

                rew = reward_np(y[i], a[0][0], past_a[0][0])
                accum_rewards += rew
                past_a = a

                simple_rewards.append(rew)

                print(str(i), ' action predicted: ', str(a[0][0]), ', reward: ', str(rew), ', accumulated reward: ', str(accum_rewards))

        for k in range(config['epochs']):
            rew_epochs[k] /= config['epochs']

        return simple_rewards, actions_returned, dates_o, instrument_o, rew_epochs

    def weights_and_biases(self):
        Ws = []
        bs = []

        ls = config['n_layers']
        l = len(config['n_layers'])

        for i in range(1,l):
            dim = ls[i-1]*self.n_features
            Ws.append(tf.Variable(tf.random_uniform([dim, dim], 0, 1)))
            bs.append(tf.Variable(tf.zeros([dim])))

        Ws.append(tf.Variable(tf.random_uniform([ls[l-1]*self.n_features+1, 1], 0, 1)))
        bs.append(tf.Variable(tf.zeros([1])))

        return Ws, bs

    def action(self, input_standard, action, Ws, bs):
        l = len(config['n_layers'])
        layer = input_standard
        for i in range(1, l):
            layer = constants.f(tf.add(tf.matmul(layer, Ws[i-1]), bs[i-1]))

        layer_augmented = tf.concat((layer, action), axis=1)
        a = constants.f(tf.add(tf.matmul(layer_augmented, Ws[l-1]), bs[l-1]))

        return a