import os
import numpy as np
import yaml

path = os.path.abspath(os.path.dirname(__file__))
filename = os.path.join(path, "../config/config.yaml")
config = yaml.load(open(filename, 'r'))

filename_price_actions_rewards = os.path.join(path, "../results/" + config['instrument'] + '_' + str(config['c']) + '_' + str(config['n_layers']) + '.csv')

def sharpe_c(rewards, window_size):
    sharpe = np.zeros(len(rewards))

    for i in range(0, len(rewards)):
        mu = np.mean(rewards[max(i-window_size,0):(i+1)])
        sigma = np.std(rewards[max(i-window_size,0):(i+1)])
        sharpe[i] = 0 if sigma == 0 else mu / sigma

    return sharpe

def read_simple_rewards_commissions(commission_name):
    filename_commission_analysis = os.path.join(path,"../results/commission_analysis_" + config['instrument'] + '_' + commission_name + '_' + str(config['n_layers']) + '.csv')

    decission_changes = []
    with open(filename_commission_analysis, "r") as file:
        for line in file:
            values = line.split(",")
            decission_changes.append(values[0])

    return decission_changes

def read_price_actions_rewards():
    dates = []
    data = []
    rewards = []
    sharpe = []
    decisions = []

    with open(filename_price_actions_rewards, "r") as file:
        for line in file:
            values = line.split(",")
            dates.append(values[0])
            data.append(values[1])
            rewards.append(values[2])
            sharpe.append(values[3])
            decisions.append(values[4])

    return dates, data, rewards, sharpe, decisions

def write(dates_test, data, simple_rewards, decisions):

    filename_commission_analysis = os.path.join(path,"../results/commission_analysis_" + config['instrument'] + '_' + str(config['c'])+ '_' + str(config['n_layers']) + '.csv')

    with open(filename_commission_analysis, "w") as file:
        for i in range(1,len(decisions)):
            file.write(str(float(abs(decisions[i]-decisions[i-1]))) + "\n")

    rewards = []
    sum = 0
    for r in simple_rewards:
        sum += r
        rewards.append(sum)

    sharpe = sharpe_c(rewards, config['window_size'])
    padding = np.zeros(len(data) - len(rewards))
    rewards = np.append(padding, rewards)
    sharpe = np.append(padding, sharpe)
    decisions = np.append(padding, decisions)

    with open(filename_price_actions_rewards, "w") as file:
        for i in range(len(data)):
            file.write(dates_test[i] + "," +
                       str(float(data[i])) + "," +
                       str(float(rewards[i])) + "," +
                       str(float(sharpe[i])) + "," +
                       str(float(decisions[i])) + "\n")

