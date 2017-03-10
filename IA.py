import random
import timeit

from theano.gradient import np

from nn import *


class IA:
    observe = 500
    buffer = 50000
    batchSize = 50
    divide_epsilon = 1000
    NUM_INPUT = 9
    GAMMA = 0.25

    def __init__(self, name, symbol, isDummy=False, isHuman = False):
        #--------
        # Game
        #--------
        self.name = name
        self.symbol = symbol
        self.isDummy = isDummy
        self.isHuman = isHuman
        self.nbWin = 0

        #--------
        # IA
        #--------
        self.tmp_state = None
        self.tmp_action = None
        self.filename = "morpion_IA_data_collect"
        self.data_collect = []
        self.loss_log = []
        self.epsilon = 0.5
        self.replay = []
        self.nn_param = [16]
        if isHuman != True and isDummy != True:
            #self.model = neural_net(self.NUM_INPUT, self.nn_param, "./h5/11000.h5")
            self.model = neural_net(self.NUM_INPUT, self.nn_param)

    def play(self, playable_position, state = None, total_game = None):

        self.tmp_state = np.asarray(state).reshape(1, self.NUM_INPUT)

        # Choose an action.
        randomNb = random.random()
        if self.isHuman:
            print(playable_position)
            self.tmp_action = int(input("index"))  # random
        elif self.isDummy:
            self.tmp_action = playable_position[random.randint(0, len(playable_position) - 1)]
        elif total_game < self.observe :
            if randomNb < 0.15 :
                self.tmp_action = random.randint(0, 8)# random
            else:
                self.tmp_action = playable_position[random.randint(0, len(playable_position) - 1)]
        elif randomNb < self.epsilon :
            self.tmp_action = playable_position[random.randint(0, len(playable_position) - 1)]
        else:
            # Get Q values for each action.
            qval = self.model.predict(self.tmp_state, batch_size=1)
            self.tmp_action = (np.argmax(qval))  # best
        return self.tmp_action

    def callbackGameStateChange(self, reward, new_state, total_frame):
        if self.isDummy != True and self.isHuman != True:
            # Take action, observe new state and get our treat.
            self.currentHitNumber = total_frame
            new_state = np.asarray(new_state).reshape(1, self.NUM_INPUT)
            # Experience replay storage.
            self.replay.append((self.tmp_state, self.tmp_action, reward, new_state))

            # If we're done observing, start training.
            if total_frame > self.observe:

                # If we've stored enough in our buffer, pop the oldest.
                if len(self.replay) > self.buffer:
                    self.replay.pop(0)

                # Randomly sample our experience replay memory
                minibatch = random.sample(self.replay, self.batchSize)

                # Get training values.
                X_train, y_train = process_minibatch(minibatch, self.model, self.GAMMA, self.NUM_INPUT)
                # Train the model on this batch.
                self.loss_log.append(self.model.train_on_batch(
                    X_train, y_train
                ))


            # Decrement epsilon over time.
            if self.epsilon > 0.1 and total_frame > self.observe:
                self.epsilon -= (1 / self.divide_epsilon)

            # Save the model every 25 000 frames.
            if total_frame % 1000 == 0 and total_frame > 0:
                self.model.save_weights("h5/"+str(total_frame+10000) + '.h5',
                                        overwrite=True)
                print("Saving model %s - %d" % (self.filename, total_frame))

    def win(self, nbPlay):
        self.nbWin += 1
        return 100

    def loose(self):
        self.nbWin = 0
        return -500

    def equal(self):
        self.nbWin = 0
        return -100

    def log_results(self):
        # Save the results to a file so we can graph it later.
        with open(self.filename + '.csv', 'w') as data_dump:
            wr = csv.writer(data_dump)
            wr.writerows(self.data_collect)

        with open(self.filename + '.csv', 'w') as lf:
            wr = csv.writer(lf)
            for loss_item in self.loss_log:
                wr.writerow(loss_item)

