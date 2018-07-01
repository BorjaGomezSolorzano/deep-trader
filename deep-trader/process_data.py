import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Read:

    def __init__(self, instrument, features_idx, instrument_idx, trainPctg, look_back):
        self.instrument = instrument
        self.features_idx = features_idx
        self.features_len = len(features_idx)
        self.instrument_idx = instrument_idx
        self.trainPctg = trainPctg
        self.look_back = look_back


    def get_data(self):
        filename = 'C:/Users/Borja/workspace_thesis/data/'+ self.instrument +'.csv'
        df = pd.read_csv(filename, names=['time', 'base', 'quote', 'instrument'], skiprows=1)
        df = df.tail(10)
        df=df.drop('time', axis=1)
        return df

    # convert an array of values into a dataset matrix
    def create_dataset(self, dataset):
        dataX, dataY = [], []
        for i in range(len(dataset) - self.look_back - 1):
            a = dataset[i:(i + self.look_back), self.features_idx]
            dataX.append(a)
            dataY.append(dataset[i + self.look_back, self.instrument_idx])
        return np.array(dataX), np.array(dataY)

    # set lastday closed as y
    def load_data(self, dataset):
        dataset_len = len(dataset)
        train_size = int(dataset_len * self.trainPctg)
        test_size = dataset_len - train_size
        train, test = dataset[0:train_size, :], dataset[train_size:dataset_len, :]

        X_train, y_train = self.create_dataset(train)
        X_test, y_test = self.create_dataset(test)

        X_train = np.reshape(X_train, (X_train.shape[0], self.look_back, self.features_len))
        X_test = np.reshape(X_test, (X_test.shape[0], self.look_back, self.features_len))

        return [X_train, y_train, X_test, y_test]

    def process(self):
        df = self.get_data()
        dataset = df.values

        # Normalize instrument values
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        dataset[:, self.instrument_idx] = self.scaler.fit_transform(dataset[:, self.instrument_idx])

        trainX, trainY, testX, testY = self.load_data(dataset)

        return trainX, trainY, testX, testY;

    def denormalize(self, data):
        return self.scaler.inverse_transform(data)