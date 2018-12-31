from model import *

def reward_tf(u, action_t, action_t1):
    return u * action_t - c * tf.abs(action_t - action_t1)

def reward_np(u, action_t, action_t1):
    return u * action_t - c * np.abs(action_t - action_t1)

def sharpe(returns):
    mu = 0
    for i in range(0, window_size):
        mu += returns[i]

    mu /= float(window_size)

    sigma = 0
    for i in range(0, window_size):
        sigma += (mu - returns[i]) * (mu - returns[i])
    sigma = (sigma / float(window_size)) ** (0.5)

    return 0 if sigma == 0 else mu / sigma


def utility(returns):
    return sharpe(returns)