import tensorflow as tf
x = tf.placeholder(tf.float32, [None, 784])

#layer 1
W1 = tf.Variable(tf.zeros([784, 100]))
b1 = tf.Variable(tf.zeros([100]))
y1 = tf.matmul(x, W1) + b1 #remove softmax

#layer 2
W2 = tf.Variable(tf.zeros([100, 10]))
b2 = tf.Variable(tf.zeros([10]))
y2 = tf.nn.softmax(tf.matmul(y1, W2) + b2)

#output
y = y2
y_ = tf.placeholder(tf.float32, [None, 10])

r=1