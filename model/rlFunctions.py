import numpy as np
import tensorflow as tf
import os
import yaml

path = os.path.abspath(os.path.dirname(__file__))
filename = os.path.join(path, "../config/config.yaml")
config = yaml.load(open(filename, 'r'))


def reward_tf(self, u, c, action_t, action_t1):
    return (u * action_t - c * tf.abs(action_t - action_t1))

def reward_np(self, u, c, action_t, action_t1):
    return (u * action_t - c * np.abs(action_t - action_t1))

def sharpe(self, returns):
    """

    :param returns:
    :return:
    """

    mu = 0
    size = config['window_size']
    for i in range(0, size):
        mu += returns[i]

    mu /= float(size)

    sigma = 0
    for i in range(0, size):
        sigma += (mu - returns[i]) * (mu - returns[i])
    sigma = (sigma / float(size)) ** (0.5)

    return 0 if sigma == 0 else mu / sigma


def utility(self, returns):
    """

    :param returns:
    :return:
    """
    return self.sharpe(returns)