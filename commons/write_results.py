import os
import numpy as np

path = os.path.abspath(os.path.dirname(__file__))


def sharpe_c(rewards):
    sharpe = np.zeros(len(rewards))
    sharpe[0] = rewards[0]

    for i in range(1, len(rewards)):
        mu = np.mean(rewards[:i + 1])
        sigma = np.std(rewards[:i + 1])

        if sigma == 0:
            continue

        sharpe[i] = mu / sigma

    return sharpe

def read(config):
    dates = []
    data = []
    rewards = []
    sharpe = []
    decisions = []

    filename = os.path.join(path, "../results/" + config['instrument'] + '_' + str(config['c']) + '.csv')

    with open(filename, "r") as file:
        for line in file:
            values = line.split(",")
            dates.append(values[0])
            data.append(values[1])
            rewards.append(values[2])
            sharpe.append(values[3])
            decisions.append(values[4])

    return dates, data, rewards, sharpe, decisions

def write(config, dates_test, data, rewards, decisions):

    sharpe = sharpe_c(rewards)
    padding = np.zeros(len(data) - len(rewards))
    rewards = np.append(padding, rewards)
    sharpe = np.append(padding, sharpe)
    decisions = np.append(padding, decisions)

    filename = os.path.join(path, "../results/" + config['instrument'] + '_' + str(config['c']) + '.csv')
    with open(filename, "w") as file:
        for i in range(len(data)):
            file.write(str(int(dates_test[i])) + "," +
                       str(float(data[i])) + "," +
                       str(float(rewards[i])) + "," +
                       str(float(sharpe[i])) + "," +
                       str(float(decisions[i])) + "\n")