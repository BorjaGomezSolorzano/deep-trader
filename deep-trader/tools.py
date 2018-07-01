# WEIGHTS AND BIASES
std_ = 1. / np.sqrt(n_neuron + 0.0)  # n_neuron / 3.
m_ = 0.
# weight and biases of the network
W_x = [tf.Variable(tf.random_normal([look_back, n_neuron], m_, 1. / np.sqrt(look_back + 0.0)))]
for _ in range(1, look_back):
    W_x.append(tf.Variable(tf.random_normal([n_neuron, n_neuron], m_, std_)))
B_x = [tf.Variable(tf.random_normal([n_neuron], m_, 1.))]
for _ in range(1, look_back):
    B_x.append(tf.Variable(tf.random_normal([n_neuron], m_, std_)))
W_m = tf.Variable(tf.random_normal([n_neuron, n_neuron], m_, std_))
W_out = tf.Variable(tf.random_normal([n_neuron, 1], m_, std_))
B_out = tf.Variable(tf.random_normal([1], m_, std_))

'''
h = f(tf.matmul(input[0], W_x[0]) + B_x[0])
action_out = np.zeros((1, 1))
for i in range(1, look_back):
    h = f(tf.matmul(h, W_x[i]) + B_x[i])
action_temp = function.action(h, W_out, B_out)
'''