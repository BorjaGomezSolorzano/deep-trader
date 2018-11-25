import numpy as np
import tensorflow as tf

class Functions:

    # REWARD
    @classmethod
    def reward_tf(self, u, c, action_t, action_t1):
        return u * action_t1 - c * tf.abs(u) * tf.abs(action_t - action_t1)

    @classmethod
    def reward_np(self, u, c, action_t, action_t1):
        return u * action_t1 - c * np.abs(u) * np.abs(action_t - action_t1)

    @classmethod
    def sharpe(self, returns):
        """

        :param returns:
        :return:
        """

        mu = 0
        size = len(returns)
        for i in range(21, size):
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
        for i in range(21, size):
            mu += returns[i]

        mu /= float(size)

        sigma = 0
        for i in range(0, size):
            sigma += (mu - returns[i]) * (mu - returns[i])
        sigma = (sigma / float(size)) ** (0.5)

        return 0 if sigma == 0 else mu / sigma

    # UTILITY function: Sharpe ratio
    @classmethod
    def sharpe1(self, returns):
        """

        :param returns:
        :return:
        """

        eta = 0.01

        sharpe_avg = 0
        size = len(returns)
        for i in range(21, size):
            print('Iteration: ', str(i))
            mu = 0
            for j in range(i-21,i):
                mu += returns[j]
            mu /= float(size)

            sigma = 0
            for j in range(i-21,i):
                sigma += (mu - returns[j]) * (mu - returns[j])
            sigma = (sigma / float(size)) ** (0.5)

            sharpe = mu / sigma

            if i == 21:
                sharpe_avg = sharpe
            else:
                sharpe_avg = sharpe * eta + (1-eta)*sharpe_avg

        return sharpe_avg


    @classmethod
    def utility(self, returns):
        """

        :param returns:
        :return:
        """
        return self.sharpe(returns)