from model import *

def place_holders():
    input_phs = []
    output_phs = []
    for i in range(window_size):
        dim = n_layers[0] * n_features
        input_phs.append(tf.placeholder(float_type_tf, shape=[1, dim]))
        output_phs.append(tf.placeholder(float_type_tf, shape=()))

    action_ph = tf.placeholder(constants.float_type_tf, shape=(1, 1))

    return input_phs, output_phs, action_ph

def recurrent_model(Ws, bs, input_ph, output_phs, action_ph):
    rewards_train = []
    a_t1 = action_ph
    for t in range(window_size):
        a_t = action(input_ph[t], a_t1, Ws, bs)
        r_t = reward_tf(output_phs[t], a_t[0][0], a_t1[0][0])
        rewards_train.append(r_t)
        a_t1 = a_t

    u = utility(rewards_train)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(-u)

    return a_t, u, optimizer