import os
import numpy as np
import yaml
from model.rlFunctions import sharpe

path = os.path.abspath(os.path.dirname(__file__))
filename = os.path.join(path, "../config/config.yaml")
config = yaml.load(open(filename, 'r'))

filename_price_actions_rewards = os.path.join(path, "../results/" + config['instrument'] + '_' + str(config['c']) + '_' + str(config['n_layers']) + '.csv')
filename_epochs = os.path.join(path,"../results/convergence_" + config['instrument'] + '_' + str(config['c']) + '_' + str(config['n_layers']) + '.csv')


def sharpe_c(rewards):

    window_size = config['window_size']
    sharpe_reaults = np.zeros(len(rewards))

    for i in range(0, window_size):
        sharpe_reaults[i] = 0

    for i in range(window_size, len(rewards)):
        sharpe_reaults[i] = sharpe(rewards[(i-window_size):(i+1)])

    return sharpe_reaults

def read_convergence():
    decission_changes = []
    with open(filename_epochs, "r") as file:
        for line in file:
            values = line.split(",")
            decission_changes.append(values[0])

    return decission_changes

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

def write(dates_test, data, simple_rewards, decisions, rew_epochs):

    with open(filename_epochs, "w") as file:
        for rew_epoch in rew_epochs:
            file.write(str(rew_epoch) + "\n")

    filename_commission_analysis = os.path.join(path,"../results/commission_analysis_" + config['instrument'] + '_' + str(config['c'])+ '_' + str(config['n_layers']) + '.csv')

    with open(filename_commission_analysis, "w") as file:
        for i in range(1,len(decisions)):
            file.write(str(float(abs(decisions[i]-decisions[i-1]))) + "\n")

    rewards = []
    sum = 0
    for r in simple_rewards:
        sum += r
        rewards.append(sum)

    sharpe = sharpe_c(simple_rewards)
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

