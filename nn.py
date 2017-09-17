"""
The design of this comes from here:
http://outlace.com/Reinforcement-Learning-Part-3/
"""

from keras.layers.core import Dense, Activation
from keras.models import Sequential
from keras.optimizers import RMSprop
import numpy as np



def neural_net(num_sensors, load=''):
    model = Sequential()

    # First layer.
    model.add(Dense(
        500, input_shape=(num_sensors,)
    ))
    model.add(Activation('relu'))  # f(input_neuron)=\max(0, input_neuron)

    # Output layer.
    model.add(Dense(9, input_shape=(500,)))
    model.add(Activation('linear'))

    rms = RMSprop()
    model.compile(loss='mse', optimizer=rms)

    if load:
        model.load_weights(load)
    return model


def process_minibatch(minibatch, model, GAMMA, NUM_INPUT, non_terminal_reward):
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

        if reward_m == non_terminal_reward: # If it's a non terminal state
            # Get prediction on new state.
            newQ = model.predict(new_state_m, batch_size=1)

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
