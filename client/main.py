import sys
sys.path.append('../')

import os
os.environ['TZ'] = 'utc'

from model import *

logger = get_logger(os.path.basename(__file__))

def read_csv():

  dates, data, rewards, sharpe, decisions = read_price_actions_rewards()

  price_rewards_actions_utility_plot(True, dates, data, rewards, decisions, sharpe)

  rew_epochs = read_convergence()

  convergence_plot(rew_epochs)


def write_csv():

  X, y, dates, instrument = process()

  rewards, actions, dates_o, instrument_o, rew_epoch = execute(X, y, dates, instrument)

  write(dates_o, instrument_o, rewards, actions, rew_epoch)

def plot_commissions():
  decission_changes_02 = read_simple_rewards_commissions('2.5e-05')
  decission_changes_05 = read_simple_rewards_commissions('0.000125')
  decission_changes_1 = read_simple_rewards_commissions('0.00025')

  commission_analysis(decission_changes_02, decission_changes_05, decission_changes_1)

if __name__ == '__main__':

  write_csv()
  read_csv()

  #plot_commissions()
