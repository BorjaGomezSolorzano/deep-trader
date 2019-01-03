from model import *

def flat(x1):
    l1 = n_layers[0]
    l = l1 * n_features
    x = np.zeros((1, l))
    for k in range(n_features):
        for j in range(l1):
            x[0][k * l1 + j] = x1[j][k]

    return x

def process():
    df = pd.read_csv(instrument_filename, skiprows=1)
    df['prices_diff'] = df.iloc[:,instrument_idx].diff(periods=1)
    df['prices_diff'] = df['prices_diff'].shift(-1)
    df = df[pd.notnull(df['prices_diff'])]

    df = df.tail(last_n_values)

    dataset = df.values

    dates = dataset[:, 0]
    instrument = np.copy(dataset[:, instrument_idx])

    returns_idx = dataset.shape[1] - 1
    X_aux, y_aux = [], []
    for i in range(len(dataset)):
        X_aux.append(dataset[i, features_idx])
        y_aux.append(dataset[i, returns_idx])

    X = np.array(X_aux, dtype=float_type_np)
    y = np.array(y_aux, dtype=float_type_np)

    return X, y, dates, instrument