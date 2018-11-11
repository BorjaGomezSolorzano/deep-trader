# -*- coding: utf-8 -*-
"""
Created on 24/09/2018

@author: Borja
"""

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from commons import constants

dirname = os.path.dirname(__file__)

class Feeder:

    def __init__(self, config):
        self.config = config
        self.filename = os.path.join(dirname, '../data/' + config['instrument'] + '.csv')
        self.features_len = len(config['features_idx'])

    # convert an array of values into a dataset matrix
    def create_dataset(self, dataset):
        returns_idx = dataset.shape[1]-1
        dataX, dataY = [], []
        for i in range(len(dataset) - 1):
            dataX.append(dataset[i, self.config['features_idx']])
            dataY.append(dataset[i, returns_idx])

        return np.array(dataX, dtype=constants.float_type_np), np.array(dataY, dtype=constants.float_type_np)

    def process(self):
        df = pd.read_csv(self.filename, skiprows=1)
        df['prices_diff'] = df.iloc[:,self.config['instrument_idx']].diff(periods=1)
        df['prices_diff'] = df['prices_diff'].shift(-1)
        df = df[pd.notnull(df['prices_diff'])]

        df = df.tail(self.config['last_n_values'])

        dataset = df.values

        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        dataset[:, self.config['features_idx']] = self.scaler.fit_transform(dataset[:, self.config['features_idx']])

        X, y = self.create_dataset(dataset[:, :])

        dates = dataset[:, 0]
        instrument = dataset[:, self.config['instrument_idx']]

        return X, y, dates, instrument

    def denormalize(self, data):
        return self.scaler.inverse_transform(data)