# -*- coding: utf-8 -*-
"""
Created on 24/09/2018

@author: Borja
"""

import os

import numpy as np
import pandas as pd
import yaml
from commons import constants

dirname = os.path.abspath(os.path.dirname(__file__))
filename = os.path.join(dirname, "../config/config.yaml")
config = yaml.load(open(filename, 'r'))

filename = os.path.join(dirname, '../data/' + config['instrument'] + '.csv')
features_len = len(config['features_idx'])

def flat(self, X_transformed, i):
    x1 = X_transformed[((i + 1) - config['n_layers'][0]):(i + 1)]
    l1 = config['n_layers'][0]
    l = l1 * self.n_features
    x = np.zeros((1, l))
    for k in range(self.n_features):
        for j in range(l1):
            x[0][k * l1 + j] = x1[j][k]

    return x

def process(self):
    df = pd.read_csv(self.filename, names=["Time","Open","Max","Min","Close","Volume"], skiprows=1)
    df['prices_diff'] = df.iloc[:,config['instrument_idx']].diff(periods=1)
    df['prices_diff'] = df['prices_diff'].shift(-1)
    df = df[pd.notnull(df['prices_diff'])]

    df = df.tail(config['last_n_values'])

    dataset = df.values

    dates = dataset[:, 0]
    instrument = np.copy(dataset[:, config['instrument_idx']])

    returns_idx = dataset.shape[1] - 1
    X_aux, y_aux = [], []
    for i in range(len(dataset)):
        X_aux.append(dataset[i, config['features_idx']])
        y_aux.append(dataset[i, returns_idx])

    X, y = np.array(X_aux, dtype=constants.float_type_np), np.array(y_aux, dtype=constants.float_type_np)

    return X, y, dates, instrument