import tensorflow as tf
import numpy as np

class Functions:

    # ACTION
    @classmethod
    def action(self, X, W_x, B_x):
        """

        :param X:
        :param W_x:
        :param B_x:
        :return:
        """
        return tf.tanh(tf.matmul(X, W_x) + B_x)

    # REWARD
    @classmethod
    def reward(self, u, c, z_t, z_tm1):
        return u * z_tm1 - c * tf.abs(z_t - z_tm1)

    def reward_array(self, u, c, z_t, z_tm1):
        return u * z_tm1 - c * np.abs(z_t - z_tm1)

    # UTILITY function: Sharpe ratio
    @classmethod
    def sharpe(self, returns):
        """

        :param returns:
        :return:
        """
        mu = 0
        sigma = 0
        size = len(returns)
        for i in range(0, size):
            mu = mu + returns[i]
        mu = mu / size

        for i in range(0, size):
            sigma = sigma + (mu - returns[i]) * (mu - returns[i])
        sigma = (sigma / size) ** (1 / 2)

        return mu / sigma

    @classmethod
    def utility(self, returns):
        """

        :param returns:
        :return:
        """
        return self.sharpe(returns)