import tensorflow as tf


class Functions:

    # ACTION
    def action(self, X, W_x, B_x):
        return tf.tanh(tf.matmul(X, W_x) + B_x)

    # REWARD
    def reward(self, u, c, z_t, z_tm1):
        return u * z_t - c * tf.abs(z_t - z_tm1)

    # UTILITY function: Sharpe ratio
    def sharpe(self, returns):
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

    def utility(self, returns):
        return self.sharpe(returns)