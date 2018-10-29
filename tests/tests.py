import tensorflow as tf
import commons.constants

x = tf.placeholder(commons.constants.float_type_tf, shape=[3, 1])
action = tf.placeholder(tf.float32, shape=(), name="action")

x1 = tf.concat([x, tf.reshape(action,(1,1))], 0)

r=1