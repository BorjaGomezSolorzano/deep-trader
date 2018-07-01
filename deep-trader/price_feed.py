import numpy as np

class Feeder:

    '''
        def generate_prices(self):
        # Generate prices to train
        input_train = []
        output_train = []
        for b in range(n_layer, len(z_train)):
            aux = []
            for a in range(b - n_layer, b):
                aux.append(z_train.iloc[a, 0])
            input_train.append(aux)
            output_train.append([z_train.iloc[b, 0]])

        input_train = np.array(input_train)
        input_train = input_train.reshape((input_train.shape[0], input_train.shape[1]))

        z_derivatives_train = np.array(output_train)
        z_derivatives_train = z_derivatives_train.reshape((z_derivatives_train.shape[0], z_derivatives_train.shape[1]))
    '''

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