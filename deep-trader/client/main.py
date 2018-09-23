import numpy as np
from feeder.price_feed import Feeder
import math
from model import Model
from agent import Agent
import tensorflow as tf
import yaml
import os

dirname = os.path.abspath(os.path.dirname(__file__))
filename = os.path.join(dirname, "../../config/config.yaml")

config = yaml.load(open(filename, 'r'))

f = tf.nn.tanh  # activation function

########################################################################

feeder = Feeder(config['instrument'], config['features_idx'], config['instrument_idx'], config['trainPctg'])
i_train, o_train, i_test, o_test = feeder.process()
print('X dimensions', i_train.shape, 'Y dimensions', o_train.shape)

series_train_length = i_train.shape[0]
series_test_length=i_test.shape[0]

n_batch = math.floor(series_train_length / config['batch_size']) + 1

model = Model(series_train_length,
              len(config['features_idx']),
              config['learning_rate'],
              config['c'],
              config['n_layers'],
              f)

optimizer, u, i_train_ph, o_train_ph, f_a_ph, Ws, bs = model.get_model()

agent = Agent(len(config['features_idx']), config['n_layers'], f)

batch_size = config['batch_size']
init = tf.global_variables_initializer()
with tf.Session() as sess:
  init.run()
  #Train
  for ep in range(config['epochs']):
    m_reward = 0
    _current_action = np.zeros((1, 1))
    for batch in range(n_batch):
      feed_dict = {}
      feed_dict[f_a_ph] = _current_action
      for i in range(series_train_length):
        feed_dict[i_train_ph[i]] = i_train[batch * batch_size:(batch + 1) * batch_size, ]
        feed_dict[o_train_ph[i]] = o_train[batch * batch_size:(batch + 1) * batch_size, ]
      reward, _ = sess.run([u, optimizer], feed_dict=feed_dict)
      m_reward += reward

  Ws_real, bs_real = sess.run([Ws, bs])

  i_test_ph = tf.placeholder(tf.float32, shape = [None, len(config['features_idx'])])
  f_a_ph = tf.placeholder(tf.float32, [1, 1])
  c_a = f_a_ph

  action = agent.get_action(i_test_ph, c_a, Ws_real, bs_real)

  # Test
  a = np.zeros((1, 1), dtype = np.float32)
  rew = 0
  next_action = 0
  actions = []
  rewards = []
  for i in range(series_test_length):
    feed_dict = {}
    feed_dict[i_test_ph] = i_test[i:i + config['n_layers'][0]]
    feed_dict[f_a_ph] = a

    a = sess.run(action, feed_dict=feed_dict)

    actions.append(a)

    # d=denorm(z_test[0,n_layer+i-1]) * out_temp - c*abs(out_temp - old_out)
    d = 1  # output_test[n_layers[0] + i - 1] * out_temp - c * abs(out_temp - old_out)
    # print(d,", ",out_temp, ",", old_out)
    rew += d
    old_action = a

    rewards.append(rew)
    # print(i," Action: ", a, "reward:", rew)