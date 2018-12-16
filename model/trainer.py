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


    def execute(self, X, y, dates, instrument):

        tf.set_random_seed(1)

        input_ph = []
        output_ph = []
        for i in range(config['window_size']):
            input_ph.append(tf.placeholder(constants.float_type_tf, shape=[config['n_layers'][0] * self.n_features, 1]))
            output_ph.append(tf.placeholder(constants.float_type_tf, shape=()))

        rewards_train = []
        action_train_ph = tf.placeholder(constants.float_type_tf, shape=(1,1))
        past_action_train = action_train_ph
        for i in range(config['window_size']):
            action_train = self.get_action(input_ph[i], past_action_train)
            rewards_train.append(reward_tf(output_ph[i], action_train[0][0], past_action_train[0][0]))
            past_action_train = action_train

        u = utility(rewards_train)
        optimizer = tf.train.AdamOptimizer(learning_rate=config['learning_rate']).minimize(-u)

        self.scaler = MinMaxScaler(feature_range=(-1, 1))

        X_transformed = self.scaler.fit_transform(np.copy(X[0:(config['n_layers'][0] + config['window_size'] + config['n_actions']), ]))

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
                        x=X_transformed[((i+1) - config['n_layers'][0]):(i+1)].reshape((config['n_layers'][0]*self.n_features,1))
                        feed_dict[input_ph[index]] = x
                        feed_dict[output_ph[index]] = y[i]

                    acum_reward, _, a, rr = sess.run([u, optimizer, action_train, rewards_train], feed_dict=feed_dict)
                    rew_epochs[ep] += acum_reward
                    #s_r=sharpe(rr)
                    #print(str(acum_reward),str(s_r),str(a),str(rr))

                i+=1
                
                dates_o.append(dates[i])
                instrument_o.append(instrument[i])

                #Test
                actions_returned.append(a[0][0])

                rew = reward_np(y[i], a[0][0], past_a[0][0])
                accum_rewards += rew
                past_a = a

                simple_rewards.append(rew)

                print(str(i), ' action predicted: ', str(a[0][0]), ', reward: ', str(rew), ', accumulated reward: ', str(accum_rewards))

        for k in range(config['epochs']):
            rew_epochs[k] /= config['epochs']

        return simple_rewards, actions_returned, dates_o, instrument_o, rew_epochs


    def get_action(self, input_standard, action):
        l = len(config['n_layers'])

        layer = tf.transpose(input_standard)
        for i in range(1, l):
            layer = tf.layers.dense(layer, units=config['n_layers'][i], activation=constants.f)

        concat = tf.concat((layer, action), axis=1)
        output_augmented = tf.layers.dense(concat, units=1, activation=constants.f)

        return output_augmented