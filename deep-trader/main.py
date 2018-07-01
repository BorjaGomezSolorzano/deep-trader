import numpy as np
import tensorflow as tf

# input parameters
instrument = 'GBPAUD'
look_back = 1
features_idx = [0,1,2] # features = ['base', 'quote']
features_len = len(features_idx)
instrument_idx = [2] # instrument = ['instrument']
trainPctg=0.9

from process_data import Read

reader = Read(instrument, features_idx, instrument_idx, trainPctg, look_back)
input_train, output_train, input_test, output_test = reader.process()
print('X dimensions', input_train.shape[0], input_train.shape[1], input_train.shape[2])
print('Y dimensions', output_train.shape[0])

from price_feed import Feeder

feeder = Feeder()

# x_train_batches, y_train_batches = feeder.prepare_batches(input_train, output_train, batch_size)
#x_test_batches, y_test_batches = feeder.prepare_batches(testX, testY, batch_size)

series_train_length=input_train.shape[0]
series_test_length=input_test.shape[0]

import math

# Model variables
f = tf.nn.tanh  # activation function
c = 0.001  # trading cost
learning_rate = 0.001  # learning rate training phase
dropout = 0.2
epochs=10
batch_size = 100
n_layers = [10, 5]
n_batch = math.floor(series_train_length / batch_size) + 1
n_iter = 150

from Direct_Reinforcement_Learning import Model

model = Model(series_train_length, features_len, learning_rate, look_back, c, n_layers)
optimizer, u, input_train_ph, output_train_ph = model.get_model()

h, input_test_ph, output_test_ph = model.get_test_model()

import tensorflow as tf

init = tf.initialize_all_variables()
with tf.Session() as sess:
  init.run()
  #Train
  for ep in range(epochs):
    m_rew = 0
    for batch in range(n_batch):
      feed_dict = {}
      for i in range(series_train_length):
        feed_dict[input_train_ph[i]] = input_train[batch * batch_size:(batch + 1) * batch_size, ]
        feed_dict[output_train_ph[i]] = output_train[batch * batch_size:(batch + 1) * batch_size, ]
      rew, _ = sess.run([u, optimizer], feed_dict=feed_dict)
      m_rew += rew

  #Test
  m = np.zeros((1, n_layers[1]))
  rew = 0
  old_out = 0
  decisions=[]
  rewards=[]
  for i in range(series_test_length):
    feed_dict = {}
    feed_dict[input_test_ph] = input_test[i:i + n_layers[0]]
    feed_dict[output_test_ph] = m

    m, o = sess.run([h, action], feed_dict=feed_dict)

    out_temp = o[0][0]
    decisions.append(out_temp)

    # d=denorm(z_test[0,n_layer+i-1]) * out_temp - c*abs(out_temp - old_out)
    d = output_test[n_layers[0] + i - 1] * out_temp - c * abs(out_temp - old_out)
    # print(d,", ",out_temp, ",", old_out)
    rew += d
    old_out = out_temp

    rewards.append(rew)
    # print(i," Action: ", a, "reward:", rew)