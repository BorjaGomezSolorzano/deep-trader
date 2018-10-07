import matplotlib.pyplot as plt
import numpy as np
from numpy import append, zeros, array


class Plot():

    def sharpe(self, rewards):
        sharpe = np.zeros(len(rewards))
        sharpe[0] = rewards[0]

        for i in range(1, len(rewards)):
          mu = np.mean(rewards[:i+1])
          sigma = np.std(rewards[:i+1])

          if sigma == 0:
            continue

          sharpe[i] = mu / sigma

        return sharpe

    def plot_results(self, data, rewards, decisions):
        sharpe=self.sharpe(rewards)
        padding = zeros(len(data)-len(rewards))
        rewards = append(padding, rewards)
        sharpe=append(padding, sharpe)
        decisions = append(padding, decisions)
        x_axis = range(len(data))

        # Two subplots, the axes array is 1-d
        nplots = 4
        f, axarr = plt.subplots(nplots, sharex=True, figsize=(10,15))
        serie=np.array(data)
        axarr[0].plot(x_axis, array(serie))
        axarr[0].set_ylabel('Prices')
        axarr[0].set_title('XAUUSD (No Side Information)')
        axarr[1].plot(x_axis, rewards)
        axarr[1].set_ylabel('Rewards')
        axarr[2].plot(x_axis, sharpe)
        axarr[2].set_ylabel('Utility function')
        axarr[3].plot(x_axis, decisions)
        axarr[3].set_ylabel('Decisions')

        for i in range(nplots):
          axarr[i].locator_params(axis='y', nbins=6)

        axarr[nplots-1].set_xlabel('Time step (t)')
        plt.show()