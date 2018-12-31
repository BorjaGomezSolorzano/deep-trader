from model import *

def execute(self, X, y, dates, instrument):

    tf.set_random_seed(1)

    Ws, bs = weights_and_biases()

    input_phs, output_phs, action_ph = place_holders()

    a_t, u, opt = recurrent_model(Ws, bs, input_phs, output_phs, action_ph)

    #The normalization
    self.scaler = MinMaxScaler(feature_range=(-1, 1))

    s=(n_layers[0] + window_size + n_actions)
    X_transformed = self.scaler.fit_transform(np.copy(X[0:s]))

    accum_rewards = 0
    past_a = np.zeros((1,1))
    actions_returned = []
    simple_rewards = []
    dates_o = []
    instrument_o = []
    rew_epochs = [0 for k in range(epochs)]

    init = tf.initialize_all_variables()
    with tf.Session() as sess:

        for j in range(n_layers[0], n_layers[0] + n_actions):

            sess.run(init)

            #Train
            for ep in range(epochs):
                feed_dict = {action_ph: np.zeros((1,1))}
                for index in range(window_size):
                    i = index + j
                    x = self.flat(X_transformed, i)
                    feed_dict[input_phs[index]] = x
                    feed_dict[output_phs[index]] = y[i]

                u_value, a, _ = sess.run([u, a_t, opt], feed_dict=feed_dict)
                rew_epochs[ep] += u_value

            i+=1

            # Test

            dates_o.append(dates[i])
            instrument_o.append(instrument[i])

            actions_returned.append(a[0][0])

            rew = reward_np(y[i], a[0][0], past_a[0][0])
            accum_rewards += rew
            past_a = a

            simple_rewards.append(rew)

            print('iteration', str(i),
                  ', action predicted: ', str(a[0][0]),
                  ', reward: ', str(rew),
                  ', accumulated reward: ', str(accum_rewards))

    for k in range(epochs):
        rew_epochs[k] /= epochs

    return simple_rewards, actions_returned, dates_o, instrument_o, rew_epochs