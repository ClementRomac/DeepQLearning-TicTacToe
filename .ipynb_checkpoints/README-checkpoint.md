# DeepQLearning-TicTacToe
Project done during the year 2016/2017 at Ingesup (Enimia Lab) : 
- Project :
  Discover DeepQLearning with a TicTacToe.

- Contributors :
  - Clément ROMAC
  - [Nicolas Luvison](https://github.com/Mitsichury)
  
We used a FeedForward Neural Network (implemented with Keras) with an experience replay buffer.

# How to use it
The main file to start is `game.py`.

## Playing vs AI
The current code in `game.py` allows you to play against the best trained AI.

## Training
Uncomment the training code in the main function of the `game.py`. 
We've implemented two types of training an AI :
- Against a random player
- Against itself

The training loss is monitored in Tensorboard. You can start Tensorboard with :
``` bash
tensorboard --logdir=./Graph
```