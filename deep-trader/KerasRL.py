# input parameters
instrument = 'GBPAUD'
look_back = 22
dropout = 0.2
features_idx = [0,1,2] # features = ['base', 'quote']
features_len = len(features_idx)
instrument_idx = [2] # instrument = ['instrument']
trainPctg=0.9
epochs=10
batch_size = 20
c = 0.0019

#We should start by creating a TensorFlow session and registering it with Keras.
#This means that Keras will use the session we registered to initialize all variables that it creates internally.
import tensorflow as tf
sess = tf.Session()

from keras import backend as K
K.set_session(sess)

# this placeholder will contain our input as flat vectors
input = tf.placeholder(tf.float32, shape=(None, look_back, features_len))

#We can then use Keras layers to speed up the model definition process:
from keras.layers import Dense
from keras.layers.recurrent import LSTM

# Keras layers can be called on TensorFlow tensors:
x = LSTM(4, input_shape=(look_back, features_len))(input)  # LSTM layer
preds = Dense(1,activation='tanh')(x)# output layer with 1 units and a tanh activation

#We define the placeholder for the labels, and the loss function we will use:
labels = tf.placeholder(tf.float32, shape=(None, 1))

from keras.objectives import mean_squared_error

loss = tf.reduce_mean(mean_squared_error(labels, preds))

#Let's train the model with a TensorFlow optimizer
from process_data import Read

processData = Read(instrument, features_idx, instrument_idx, trainPctg, look_back)
trainX, trainY, testX, testY = processData.process()

#Do the action/reward step
from rlFunctions import Functions

functions = Functions()

out = []
rewards = []
series_train_length = trainX.shape[0]
for t in range(series_train_length):
    action_temp = outerBlock
    if t == 0:
        rewards.append(tf.reduce_sum(functions.reward(Z_LSTM_train[0], c, action_temp, 0)))
    else:
        rewards.append(tf.reduce_sum(functions.reward(Z_LSTM_train[t], c, action_temp, action_out)))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

from price_feed import Feeder

feeder = Feeder()

x_batches, y_batches = feeder.prepare_batches(trainX, trainY, batch_size)
#x_test_batches, y_test_batches = feeder.prepare_batches(testX, testY, batch_size)

# Initialize all variables
init = tf.global_variables_initializer()
sess.run(init)

with sess.as_default():
    # Run training loop
    for ep in range(epochs):
        for x, y in zip(x_batches, y_batches):
            sess.run(train_step, feed_dict={input: x, labels: y})
            if ep % 100 == 0:
                mse = loss.eval(feed_dict={input: x, labels: y})
                #mse_test = loss.eval(feed_dict={input: x_test_batches, labels: y_test_batches})
                print("Epoch: {}/{}".format(ep, epochs), "MSE train:", mse)

    #predict
    y_pred = sess.run(preds, feed_dict={input: testX})

    print(processData.denormalize(y_pred))