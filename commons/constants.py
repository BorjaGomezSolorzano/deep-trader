import os
import yaml
import tensorflow as tf
import numpy as np

f = tf.nn.tanh  # activation function
float_type_tf = tf.float32
float_type_np = np.float32

path = os.path.abspath(os.path.dirname(__file__))
filename = os.path.join(path, "../config/config.yaml")
config = yaml.load(open(filename, 'r'))

instrument_filename = os.path.join(path, '../data/' + config['instrument'] + '.csv')

features_idx = config['features_idx']
n_features = len(config['features_idx'])
n_layers = config['n_layers']
window_size = config['window_size']
n_actions = config['n_actions']
epochs = config['epochs']
c = config['c']
learning_rate = config['learning_rate']
instrument_idx = config['instrument_idx']
last_n_values = config['last_n_values']
multiplier = config['multiplier']