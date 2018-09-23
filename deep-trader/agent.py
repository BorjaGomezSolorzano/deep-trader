
import tensorflow as tf

class Agent:

    def __init__(self,
                 features_len,
                 n_layers,
                 f):

        self.features_len = features_len
        self.n_layers = n_layers
        self.f = f

    def get_action(self, input, past_action, Ws, bs):
        """

        :param input:
        :param action:
        :param Ws:
        :param bs:
        :return:
        """
        standard_memory = tf.matmul(input, Ws[0][0:self.features_len])
        # The memory from the past
        recurrent_memory = tf.matmul(past_action ,tf.reshape(Ws[0][self.features_len], [1, self.n_layers[0]]))
        input_layer = self.f(tf.add(tf.add(standard_memory, recurrent_memory), bs[0]))
        hidden_layer = self.f(tf.add(tf.matmul(input_layer, Ws[1]), bs[1]))
        output_layer = self.f(tf.add(tf.matmul(hidden_layer, Ws[2]), bs[2]))

        return output_layer