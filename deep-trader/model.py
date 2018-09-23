
import tensorflow as tf
from rlFunctions import Functions
from agent import Agent

class Model():

    def __init__(self,
                 series_train_length,
                 features_len,
                 learning_rate,
                 c,
                 n_layers,
                 f):
        self.series_train_length = series_train_length
        self.features_len=features_len
        self.learning_rate=learning_rate
        self.c=c
        self.n_layers = n_layers
        self.f = f
        self.agent = Agent(features_len,n_layers,f)

    def get_model(self):
        """

        :return:
        """
        i_t_ph = []
        o_t_ph = []
        for _ in range(self.series_train_length):
            i_t_ph.append(tf.placeholder(tf.float32, shape=[None,self.features_len]))
            o_t_ph.append(tf.placeholder(tf.float32, shape=[None, 1]))

        functions = Functions()

        Ws = [tf.Variable(tf.random_uniform([self.features_len + 1, self.n_layers[0]]), dtype=tf.float32),
              tf.Variable(tf.random_uniform([self.n_layers[0], self.n_layers[1]]), dtype=tf.float32),
              tf.Variable(tf.random_uniform([self.n_layers[1], 1]), dtype=tf.float32)]

        bs = [tf.Variable(tf.zeros([self.n_layers[0]]), dtype=tf.float32),
              tf.Variable(tf.zeros([self.n_layers[1]]), dtype=tf.float32),
              tf.Variable(tf.zeros([1]), dtype=tf.float32)]

        rewards = []
        f_a_ph = tf.placeholder(tf.float32, shape=[1,1])
        c_a = f_a_ph
        for t in range(1, self.series_train_length):
            n_a = self.agent.get_action(i_t_ph[t], c_a, Ws, bs)
            rewards.append(tf.reduce_sum(functions.reward(o_t_ph[t], self.c, n_a, 0 if t == 0 else c_a)))
            c_a = n_a

        u = functions.utility(rewards)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(-u)

        return optimizer, u, i_t_ph, o_t_ph, f_a_ph, Ws, bs
