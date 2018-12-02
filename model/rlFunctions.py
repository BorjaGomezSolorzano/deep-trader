import numpy as np
import tensorflow as tf

class Functions:

    # REWARD
    @classmethod
    def reward_tf(self, u, c, action_t, action_t1):
        return (u * action_t - c * tf.abs(action_t - action_t1))

    @classmethod
    def reward_np(self, u, c, action_t, action_t1):
        return (u * action_t - c * np.abs(action_t - action_t1))

    @classmethod
    def sharpe(self, returns):
        """

        :param returns:
        :return:
        """

        mu = 0
        size = len(returns)
        for i in range(0, size):
            mu += returns[i]

        mu /= float(size)

        return mu

    @classmethod
    def sharpe2(self, returns):
        """

        :param returns:
        :return:
        """

        mu = 0
        size = len(returns)
        for i in range(0, size):
            mu += returns[i]

        mu /= float(size)

        sigma = 0
        for i in range(0, size):
            sigma += (mu - returns[i]) * (mu - returns[i])
        sigma = (sigma / float(size)) ** (0.5)

        return 0 if sigma == 0 else mu / sigma


    @classmethod
    def utility(self, returns):
        """

        :param returns:
        :return:
        """
        return self.sharpe2(returns)