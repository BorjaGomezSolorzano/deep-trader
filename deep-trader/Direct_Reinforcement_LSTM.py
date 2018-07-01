import numpy as np
import tensorflow as tf

# Model variables
block_layer = 3  # number of neuron each layer
block_neuron = 10
base_neuron = 10
base_layer = 1
lstm_neuron = 10
outcome_layer = 2
outcome_neuron = 10
dropout = 1.
shuffle = False
f = tf.nn.tanh
batch_size = 10
n_iter = 150
gamma = 1  # discount factor
c = 0.0019
window = 1
learning_rate=0.001 #learning rate training phase

Z_LSTM_in_train = []
for _ in range(series_train_length):
    Z_LSTM_in_train.append(tf.placeholder("float", [None, window]))

Z_LSTM_train = []
for _ in range(series_train_length):
    Z_LSTM_train.append(tf.placeholder("float", [None, 1]))

Comm_train = []
for _ in range(series_train_length):
    Comm_train.append(tf.placeholder("float", [None, 1]))


def Block(input_, input_size, neuron_list, f, variables):
    last_input = input_
    last_input_size = input_size
    for neurons in neuron_list:
        std = 1. / np.sqrt(input_size + 0.0)
        W = tf.Variable(tf.random_normal([last_input_size, neurons], 0., std))
        b = tf.Variable(tf.random_normal([neurons], 0., std))
        last_input = f(tf.matmul(last_input, W) + b)
        last_input = tf.nn.dropout(last_input, dropout)
        last_input_size = neurons
        variables.append(W)
        variables.append(b)
    return last_input


# (content of the cell, real output)
def Lstm(input_, write_, reset_, output_, last_lstm):
    lstm = input_ * write_ + reset_ * last_lstm
    return (lstm, lstm * output_)


def Merge(input_list, dim_list, out_dim, f, variables):
    sum_ = np.zeros((1, out_dim))
    for input_, dim_ in zip(input_list, dim_list):
        std = 1. / np.sqrt(dim_ + 0.0)
        W = tf.Variable(tf.random_normal([dim_, out_dim], 0., std))
        sum_ = sum_ + tf.matmul(tf.cast(input_, tf.float32), tf.cast(W, tf.float32))
        variables.append(W)
    b = tf.Variable(tf.random_normal([out_dim], 0., std))
    variables.append(b)
    return f(sum_ + b)


out = []
rewards = []

from rlFunctions import Functions

functions = Functions()

variables = []

action_out = 0.
lstm = np.zeros((1, 10))
lstm_out = np.zeros((1, 10))
for t in range(series_train_length):
    inputShared1 = Merge([Z_LSTM_in_train[t], Comm_train[t]], [window, 1], 10, tf.tanh, variables)
    sharedBlock1 = Block(inputShared1, 10, [10] * 2, tf.tanh, variables)
    inputShared2 = Merge([sharedBlock1, lstm_out], [10, 10], 10, tf.tanh, variables)
    sharedBlock2 = Block(inputShared2, 10, [10], tf.tanh, variables)
    block1 = Block(sharedBlock2, 10, [10] * 3, tf.tanh, variables)
    block2 = Block(sharedBlock2, 10, [10] * 3, tf.tanh, variables)
    block3 = Block(sharedBlock2, 10, [10] * 3, tf.tanh, variables)
    block4 = Block(sharedBlock2, 10, [10] * 3, tf.tanh, variables)
    lstm, lstm_out = Lstm(block1, block2, block3, block4, lstm)
    outerBlock = Block(lstm_out, 10, [10, 10, 1], tf.tanh, variables)

    action_temp = outerBlock
    out.append(outerBlock)

    # the commission is halfed because
    # going from 0 to 1 you pay one
    # going from 1 to 0 you pay one (but you should pay 0 because you already paid)
    # going from 0 to -1 you pay 1
    # going from -1 to 0 you pay 1 (and you should pay 0)
    # going from -1 to 1 you pay 2 (and you should pay 1)
    # goind from 1 to -1 you pay 2 (and you should pay 1)
    # so the average cost is 8./6. but it should be 4./6. (assuming each kind of transaction happens with same probability). So 8./6. * 1./2. = 4./6.

    if t == 0:
        rewards.append(tf.reduce_sum(functions.reward(Z_LSTM_train[0], c, action_temp, 0)))
    else:
        rewards.append(tf.reduce_sum(functions.reward(Z_LSTM_train[t], c, action_temp, action_out)))

    action_out = action_temp

# Utility
u = functions.utility(rewards)

# we should max u, or the same min -u
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(-u)