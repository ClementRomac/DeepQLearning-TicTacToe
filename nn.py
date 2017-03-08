"""
The design of this comes from here:
http://outlace.com/Reinforcement-Learning-Part-3/
"""
import csv

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import RMSprop
from keras.callbacks import Callback
from theano.gradient import np
from sklearn.linear_model import LogisticRegression
import os


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


'''
    Create neuronal layers and add optimizers
    ReLu -> https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
'''


def neural_net(num_sensors, params, load=''):
    model = Sequential()

    # First layer.
    # model.add(Dense(
    #     params[0], input_shape=(num_sensors,)
    # ))
    # model.add(Activation('relu'))  # f(input_neuron)=\max(0, input_neuron)

    # Output layer.
    model.add(Dense(9, input_shape=(num_sensors,)))
    model.add(Activation('linear'))

    rms = RMSprop()
    model.compile(loss='mse', optimizer=rms)

    '''RMSProp optimizer.

       It is recommended to leave the parameters of this optimizer
       at their default values
       (except the learning rate, which can be freely tuned).

       This optimizer is usually a good choice for recurrent
       neural networks.

       # Arguments
           lr: float >= 0. Learning rate.
           rho: float >= 0.
           epsilon: float >= 0. Fuzz factor.
           decay: float >= 0. Learning rate decay over each update.
       '''
    if load:
        model.load_weights(load)
    return model

    # model = LogisticRegression(warm_start=True)

def process_minibatch(minibatch, model, GAMMA, NUM_INPUT):
    """This does the heavy lifting, aka, the training. It's super jacked."""
    X_train = []
    y_train = []
    # Loop through our batch and create arrays for X and y
    # so that we can fit our model at every step.
    for memory in minibatch:
        # Get stored values.
        old_state_m, action_m, reward_m, new_state_m = memory
        # Get prediction on old state.
        old_qval = model.predict(old_state_m, batch_size=1)
        y = np.zeros((1, NUM_INPUT))
        y[:] = old_qval[:]

        if reward_m == 5:
            # Get prediction on new state.
            newQ = model.predict(new_state_m, batch_size=1)
            # Get our best move. I think?
            maxQ = np.max(newQ)

            update = (reward_m + (GAMMA * maxQ))
        else:  # terminal state
            update = reward_m
        # Update the value for the action we took.
        y[0][action_m] = update
        X_train.append(old_state_m.reshape(NUM_INPUT, ))
        y_train.append(y.reshape(NUM_INPUT, ))

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train, y_train


def log_results(filename, data_collect, loss_log):
    # Save the results to a file so we can graph it later.
    with open(filename + '.csv', 'w') as data_dump:
        wr = csv.writer(data_dump)
        wr.writerows(data_collect)

    with open(filename + '.csv', 'w') as lf:
        wr = csv.writer(lf)
        for loss_item in loss_log:
            wr.writerow(loss_item)
