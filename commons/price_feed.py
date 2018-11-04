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

    def instrument_values(self):
        df = pd.read_csv(self.filename, skiprows=1)
        df = df.tail(self.config['last_n_values'])

        dataset = df.values

        dataset_len = len(dataset)
        train_size = int(dataset_len * self.config['trainPctg'])

        dates_train = dataset[0:train_size, 0]
        instrument_train = dataset[0:train_size, self.config['instrument_idx']]
        dates_test = dataset[train_size:dataset_len, 0]
        instrument_test = dataset[train_size:dataset_len, self.config['instrument_idx']]

        return dates_train, instrument_train, dates_test, instrument_test

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

        dataset_len = len(dataset)
        train_size = int(dataset_len * self.config['trainPctg'])

        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        dataset[:, self.config['features_idx']] = self.scaler.fit_transform(dataset[:, self.config['features_idx']])
        train, test = dataset[0:train_size, :], dataset[(train_size - self.config['n_layers'][0]):dataset_len, :]

        X_train, y_train = self.create_dataset(train)
        X_test, y_test = self.create_dataset(test)
        X_train = np.reshape(X_train, (X_train.shape[0], self.features_len))
        X_test = np.reshape(X_test, (X_test.shape[0], self.features_len))

        return X_train, y_train, X_test, y_test

    def denormalize(self, data):
        return self.scaler.inverse_transform(data)

    def get_batches(self, X, y, batch_size = 10):
        """ Return a generator for batches """
        n_batches = len(X) // batch_size
        X, y = X[:n_batches * batch_size], y[:n_batches * batch_size]

        # Loop over batches and yield
        for b in range(0, len(X), batch_size):
            yield X[b:b + batch_size], y[b:b + batch_size]

    def prepare_batches(self, X, Y, batch_size=20):
        # Generate batches
        x_batches = []
        y_batches = []
        for x, y in self.get_batches(X, Y, batch_size):
            x_batches.append(x)
            y_batches.append(y)

        x_batches = np.array(x_batches)
        y_batches = np.array(y_batches)

        return x_batches, y_batches