# -*- coding: utf-8 -*-
"""
Created on 11/09/2018

@author: Borja
"""

import os

import tensorflow as tf
import yaml
from sklearn.preprocessing import MinMaxScaler
from model.reinforcemen_learning_functions import reward_np
import numpy as np

dirname = os.path.abspath(os.path.dirname(__file__))
filename = os.path.join(dirname, "../config/config.yaml")
config = yaml.load(open(filename, 'r'))


def execute(self, X, y, dates, instrument):

    tf.set_random_seed(1)

    Ws, bs = weights_and_biases()

    input_ph, output_ph, action_train_ph = place_holders()

    past_action_train, u, optimizer = recurrent_model(Ws, bs, input_ph, output_ph, action_train_ph)

    #The normalization
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