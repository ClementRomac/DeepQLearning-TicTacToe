import csv
import random
from enum import Enum
import os
import tensorflow as tf
import datetime
import pickle
import json

from nn import *


class AITypes(Enum):
    ANN = 1,
    RANDOM = 2,
    HUMAN = 3


class RewardTypes(Enum):
    WIN = 1,
    LOOSE = 2,
    DRAW = 3,
    NOTHING = 4


class AI:
    observe = 100
    buffer = 50000
    batchSize = 20
    divide_epsilon = 100000
    NUM_INPUT = 9
    GAMMA = 0.8
    epsilon = 0.7

    def __init__(self, name, symbol, AIType, load_weights=None, isTraining=False):
        # --------
        # Game
        # --------
        self.name = name
        self.symbol = symbol
        self.nbWin = 0
        self.AIType = AIType
        self.isTraining = isTraining

        # --------
        # IA
        # --------
        self.tmp_state = None
        self.tmp_action = None
        self.loss_log = []
        self.replay = []

        if AIType == AITypes.ANN:
            now = datetime.datetime.now()
            self.saving_folder = str(now.year) + "." + str(now.month) + "." + str(now.day)
            self.tb_writer = tf.summary.FileWriter("./Graph/" + self.saving_folder)
            if load_weights:
                self.model = neural_net(self.NUM_INPUT, "./h5/" + str(load_weights) + ".h5")
            else:
                self.model = neural_net(self.NUM_INPUT)

    def play(self, playable_position, state=None, total_game=None):

        self.tmp_state = np.asarray(state).reshape(1, self.NUM_INPUT)

        # Choose an action.
        randomNb = random.random()
        if self.AIType == AITypes.HUMAN:
            print(playable_position)
            self.tmp_action = int(input("index"))  # random
        elif self.AIType == AITypes.RANDOM:
            self.tmp_action = playable_position[random.randint(0, len(playable_position) - 1)]
        else:
            if self.isTraining:
                if total_game < self.observe:
                    if randomNb < 0.15:
                        self.tmp_action = random.randint(0, 8)  # random
                    else:
                        self.tmp_action = playable_position[random.randint(0, len(playable_position) - 1)]
                else:
                    if randomNb < self.epsilon:
                        self.tmp_action = playable_position[random.randint(0, len(playable_position) - 1)]
                    else:
                        # Get Q values for each action.
                        qval = self.model.predict(self.tmp_state, batch_size=1)
                        self.tmp_action = (np.argmax(qval))  # best
            else:
                qval = self.model.predict(self.tmp_state, batch_size=1)
                self.tmp_action = (np.argmax(qval))  # best

        return self.tmp_action

    def callbackGameStateChange(self, reward, new_state, total_frame):
        if self.AIType == AITypes.ANN and self.isTraining:
            # Take action, observe new state and get our treat.
            new_state = np.asarray(new_state).reshape(1, self.NUM_INPUT)
            # Experience replay storage.
            self.replay.append((self.tmp_state, self.tmp_action, reward, new_state))

            if total_frame == self.observe:
                print("----------------------------- TRAINING ----------------------------- ")

            # If we're done observing, start training.
            if total_frame > self.observe:

                # If we've stored enough in our buffer, pop the oldest.
                if len(self.replay) > self.buffer:
                    self.replay.pop(0)

                # Randomly sample our experience replay memory
                minibatch = random.sample(self.replay, self.batchSize)

                # Get training values.
                X_train, y_train = process_minibatch(minibatch, self.model, self.GAMMA, self.NUM_INPUT,
                                                     self.getReward(RewardTypes.NOTHING))
                # Train the model on this batch.
                # self.loss_log.append(self.model.fit(
                #     X_train, y_train, batch_size=self.batchSize, nb_epoch=1, verbose=0
                # ).history)

                loss = self.model.train_on_batch(X_train, y_train)

                self.loss_log.append(loss)

                summary = tf.Summary(value=[tf.Summary.Value(tag="Loss", simple_value=loss)])
                self.tb_writer.add_summary(summary, total_frame)

                # Decrement epsilon over time.
                if self.epsilon > 0.1:
                    self.epsilon -= (1 / self.divide_epsilon)

            # Save the model every 1 000 frames.
            if total_frame % 1000 == 0 and total_frame > 0 and reward != self.getReward(RewardTypes.NOTHING):
                print("Saving model and results for %s - %d" % ("Morpion AI", total_frame))
                self.log_ai_and_weights(total_frame)

    def getReward(self, rewardType):
        if rewardType == RewardTypes.WIN:
            self.nbWin += 1
            return 20
        elif rewardType == RewardTypes.LOOSE:
            self.nbWin = 0
            return -20
        elif rewardType == RewardTypes.DRAW:
            self.nbWin = 0
            return 10
        elif rewardType == RewardTypes.NOTHING:
            return 0.5

    def log_ai_and_weights(self, frames):
        #### WEIGHTS ####
        if not os.path.exists("h5/" + self.saving_folder):
            os.makedirs("h5/" + self.saving_folder + "/")
        self.model.save_weights("h5/" + self.saving_folder + "/" + str(frames) + '.h5',
                                overwrite=True)

        #### PICKLE ####
        if not os.path.exists("pickles/" + self.saving_folder):
            os.makedirs("pickles/" + self.saving_folder + "/")
        with open("pickles/" + self.saving_folder + "/" + str(frames) + '.pickle', 'wb') as ai_dump:
            dict_to_dump = {
                'epsilon': self.epsilon,
                'name': self.name,
                'symbol': self.symbol,
                'nbWin': self.nbWin,
                'AIType': self.AIType,
                'isTraining': self.isTraining,
                'replay': self.replay,
                'tmp_state': self.tmp_state,
                'tmp_action': self.tmp_action,
                'loss_log': self.loss_log
            }

            pickle.dump(dict_to_dump, ai_dump, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_ai(checkpoint):
        with open("pickles/" + str(checkpoint) + '.pickle', 'rb') as handle:
            ai_dict = pickle.load(handle)

            ai = AI(ai_dict['name'], ai_dict['symbol'], ai_dict['AIType'], load_weights=checkpoint, isTraining=ai_dict['isTraining'])
            ai.epsilon = ai_dict['epsilon']
            ai.nbWin = ai_dict['nbWin']
            ai.replay = ai_dict['replay']
            ai.tmp_state = ai_dict['tmp_state']
            ai.tmp_action = ai_dict['tmp_action']
            ai.loss_log = ai_dict['loss_log']

            return ai
