import tensorflow as tf
import numpy as np

class Functions:

    # REWARD
    @classmethod
    def reward(self, u, c, action_t, action_t1):
        return u * action_t - c * tf.abs(action_t - action_t1)

    def reward_array(self, u, c, action_t, action_t1):
        return u * action_t - c * np.abs(action_t - action_t1)

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
        #sp = np.mean(returns) / np.std(returns)
        #mu, sigma = tf.nn.moments(returns, axes=[1])

        return mu / sigma

    @classmethod
    def utility(self, returns):
        """

        :param returns:
        :return:
        """
        return self.sharpe(returns)