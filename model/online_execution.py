from model import *

def execute(X, y, dates, prices):

    tf.set_random_seed(1)

    Ws, bs = weights_and_biases()

    input_phs, output_phs, action_ph, prices_phs = place_holders()

    last_a, u, opt = recurrent_model(Ws, bs, input_phs, output_phs, action_ph,
                                     prices_phs)

    scaler = MinMaxScaler(feature_range=(-1, 1))

    accum_rewards = 0
    a_t1 = np.zeros((1,1))
    actions_returned = []
    simple_rewards = []
    dates_o = []
    instrument_o = []
    rew_epochs = [0 for _ in range(epochs)]

    init = tf.initialize_all_variables()
    with tf.Session() as sess:

        for j in range(n_layers[0], n_layers[0] + n_actions):

            sess.run(init)

            x_dic = {}
            for index in range(window_size):
                i = index + j
                x_i = np.copy(X[((i+1)-n_layers[0]):(i+1)])
                x_dic[index] = scaler.fit_transform(x_i)

            #Train
            for ep in range(epochs):
                feed_dict = {action_ph: np.zeros((1,1))}
                for index in range(window_size):
                    i = index + j
                    x = flat(x_dic[index])
                    feed_dict[input_phs[index]] = x
                    feed_dict[output_phs[index]] = y[i]
                    feed_dict[prices_phs[index]] = prices[i]

                u_value, a_t, _ = sess.run([u, last_a, opt], feed_dict=feed_dict)
                rew_epochs[ep] += u_value

            i+=1

            # Test

            dates_o.append(dates[i])
            instrument_o.append(prices[i])

            actions_returned.append(a_t[0][0])

            rew = reward_np(y[i], a_t[0][0], a_t1[0][0], prices[i])
            accum_rewards += rew
            a_t1 = a_t

            simple_rewards.append(rew)

            print('iteration', str(i),
                  ', action predicted: ', str(a_t[0][0]),
                  ', reward: ', str(rew),
                  ', accumulated reward: ', str(accum_rewards))

    for k in range(epochs):
        rew_epochs[k] /= epochs

    return simple_rewards, actions_returned, dates_o, instrument_o, rew_epochs