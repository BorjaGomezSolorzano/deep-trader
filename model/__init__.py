import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from commons import constants
from commons.constants import features_idx, last_n_values, float_type_np, instrument_idx, instrument_filename, \
    n_features, float_type_tf, window_size, learning_rate, n_layers, c, multiplier
from commons.constants import n_layers, window_size, n_actions, epochs
from commons.interactive_plots import price_rewards_actions_utility_plot, convergence_plot, commission_analysis
from commons.logUtils import get_logger
from commons.write_results import write, read_price_actions_rewards, read_convergence, read_simple_rewards_commissions
from model.price_feed import process, flat
from model.reinforcemen_learning_functions import reward_np, reward_tf, utility
from model.traders_action import action
from model.recurrent_action_model import place_holders, recurrent_model
from model.weights_and_biases import weights_and_biases
from model.online_execution import execute
