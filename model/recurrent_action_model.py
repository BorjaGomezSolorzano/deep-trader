import os

from commons import constants
import tensorflow as tf
import yaml

dirname = os.path.abspath(os.path.dirname(__file__))
filename = os.path.join(dirname, "../config/config.yaml")
config = yaml.load(open(filename, 'r'))

def place_holders():
    # The placeholders for the input and the output
    input_ph = []
    output_ph = []
    for i in range(config['window_size']):
        input_ph.append(tf.placeholder(constants.float_type_tf, shape=[1, config['n_layers'][0] * self.n_features]))
        output_ph.append(tf.placeholder(constants.float_type_tf, shape=()))

    action_train_ph = tf.placeholder(constants.float_type_tf, shape=(1, 1))

    return input_ph, output_ph, action_train_ph

def recurrent_model(Ws, bs, input_ph, output_ph, action_train_ph):

    # The recurrent action model
    rewards_train = []
    past_action_train = action_train_ph
    for i in range(config['window_size']):
        action_train = action(input_ph[i], past_action_train, Ws, bs)
        rewards_train.append(reward_tf(output_ph[i], action_train[0][0], past_action_train[0][0]))
        past_action_train = action_train

    u = utility(rewards_train)
    optimizer = tf.train.AdamOptimizer(learning_rate=config['learning_rate']).minimize(-u)

    return past_action_train, u, optimizer