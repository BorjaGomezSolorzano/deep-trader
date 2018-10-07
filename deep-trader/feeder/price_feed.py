# -*- coding: utf-8 -*-
"""
Created on 24/09/2018

@author: Borja
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import constants

class Feeder:

    def __init__(self, instrument, features_idx, instrument_idx, trainPctg):
        self.instrument = instrument
        self.features_idx = features_idx
        self.features_len = len(features_idx)
        self.instrument_idx = instrument_idx
        self.trainPctg = trainPctg

    def get_data(self):
        filename = 'C:/Users/Borja/deep-trader-master/data/' + self.instrument + '.csv'
        df = pd.read_csv(filename, names=['time', 'base', 'quote', 'instrument'], skiprows=1)
        df['prices_diff']=df['instrument'].diff(periods=1)
        df = df.drop('time', axis=1)
        df = df.tail(200)

        return df

        # convert an array of values into a dataset matrix

    def create_dataset(self, dataset):
        dataX, dataY = [], []
        for i in range(len(dataset) - 1):
            a = dataset[i, self.features_idx]
            dataX.append(a)
            dataY.append(dataset[i, self.instrument_idx])

        return np.array(dataX,dtype=constants.float_type_np), np.array(dataY,dtype=constants.float_type_np)

        # set lastday closed as y

    def load_data(self, dataset):
        dataset_len = len(dataset)
        train_size = int(dataset_len * self.trainPctg)
        train, test = dataset[0:train_size, :], dataset[train_size:dataset_len, :]

        X_train, y_train = self.create_dataset(train)
        X_test, y_test = self.create_dataset(test)

        X_train = np.reshape(X_train, (X_train.shape[0], self.features_len))
        X_test = np.reshape(X_test, (X_test.shape[0], self.features_len))

        return [X_train, y_train, X_test, y_test]

    def process(self):
        df = self.get_data()
        dataset = df.values

        # Normalize instrument values
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        dataset[:, self.features_idx] = self.scaler.fit_transform(dataset[:, self.features_idx])

        trainX, trainY, testX, testY = self.load_data(dataset)

        return trainX, trainY, testX, testY

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

        #x_batches = x_batches.reshape((x_batches.shape[0], x_batches.shape[1], 1))
        #y_batches = y_batches.reshape((y_batches.shape[0], y_batches.shape[1], 1))

        return x_batches, y_batches