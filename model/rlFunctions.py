import numpy as np


class Functions:

    # REWARD
    @classmethod
    def reward(self, u, c, action_t, action_t1):
        return u * action_t1 - c * np.abs(action_t - action_t1)

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
            mu += returns[i]
        mu /= size

        for i in range(0, size):
            sigma += (mu - returns[i]) * (mu - returns[i])
        sigma = (sigma / size) ** (1 / 2)

        return mu / sigma

    @classmethod
    def utility(self, returns):
        """

        :param returns:
        :return:
        """
        return self.sharpe(returns)