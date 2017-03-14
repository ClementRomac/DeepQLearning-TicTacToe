from IPython.core.display import display

from AI import AI
from AI import AITypes
import matplotlib.pyplot as plt

DISPLAY_INFO = False
REWARD_ONE_HIT = 1


def displayGrid(grid):
    print_line(grid)
    print_line(grid, 3)
    print_line(grid, 6)
    print(" -------------")

    print()


def print_line(grid, offset=0):
    print(" -------------")
    for i in range(3):
        if grid[i + offset] == 0:
            print(" | " + " ", end='')
        else:
            print(" | " + str(grid[i + offset]), end='')
    print(" |")


def is_win(grid):
    if (grid[0] == grid[1]) and (grid[0] == grid[2]) and (grid[0] != 0):
        return 1
    if (grid[3] == grid[4]) and (grid[3] == grid[5]) and (grid[3] != 0):
        return 1
    if (grid[6] == grid[7]) and (grid[6] == grid[8]) and (grid[6] != 0):
        return 1
    if (grid[0] == grid[3]) and (grid[0] == grid[6]) and (grid[0] != 0):
        return 1
    if (grid[1] == grid[4]) and (grid[1] == grid[7]) and (grid[1] != 0):
        return 1
    if (grid[2] == grid[5]) and (grid[2] == grid[8]) and (grid[2] != 0):
        return 1
    if (grid[0] == grid[4]) and (grid[0] == grid[8]) and (grid[0] != 0):
        return 1
    if (grid[2] == grid[4]) and (grid[2] == grid[6]) and (grid[2] != 0):
        return 1


def Stroke(state_vector_param, joueur, nbGame):
    playable_position = []
    positionIsBad = False
    i = 0
    while i < len(state_vector_param):
        if state_vector_param[i] == 0:
            playable_position.append(i)
        i += 1
    index = joueur.play(playable_position, state_vector_param, nbGame)

    if state_vector_param[index] != 0:
        positionIsBad = True

    state_vector_param[index] = joueur.symbol

    if DISPLAY_INFO:
        print(joueur.name, " (", joueur.symbol, ")", sep='')
        displayGrid(state_vector_param)

    return positionIsBad


def is_draw(grid):
    for i in range(9):
        if grid[i] == 0:
            return 0
    return 1


def playAGame(player, nbGame):
    winner = False
    nb_frames_per_game = 0
    while not winner:
        positionIsBad = Stroke(state_vector, ais[player], nbGame)
        if positionIsBad:
            print("bad position")
            if player == 1:
                return ais[2], nb_frames_per_game, state_vector
            elif player == 2:
                return ais[1], nb_frames_per_game, state_vector

        if is_win(state_vector):
            if DISPLAY_INFO:
                print("Winner: " + ais[player].name)
            return ais[player], nb_frames_per_game, state_vector
        else:
            if is_draw(state_vector):
                if DISPLAY_INFO:
                    print("It's a draw !")
                return None, nb_frames_per_game, state_vector
        ais[player].callbackGameStateChange(REWARD_ONE_HIT, state_vector, nbGame)
        if player == 1:
            player = 2
        else:
            player = 1
        nb_frames_per_game += 1


def init():
    player = 1
    state_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    return player, state_vector


if __name__ == '__main__':
    global ais
    ais = {
        1: AI("IA", 1, AITypes.ANN),
        2: AI("Random Player", 2, AITypes.RANDMOM)
    }
    #plt.xlabel('Nb of Games')
    #plt.ylabel('Human Wins')
    #plot = plt.plot(0,0)
    #plt.axis((500, 10000, 0, 50))


    for i in range(0, 20001):
        player, state_vector = init()
        winner, nbPlay, state_vector = playAGame(player, i)
        if winner == ais[1]:
            rewardWinner = ais[1].win(nbPlay)
            ais[1].callbackGameStateChange(rewardWinner, state_vector, i)
            rewardLooser = ais[2].loose()
            ais[2].callbackGameStateChange(rewardLooser, state_vector, i)
            print(ais[1].name, " wins", ais[1].nbWin)
        elif winner == ais[2]:
            rewardWinner = ais[2].win(nbPlay)
            ais[2].callbackGameStateChange(rewardWinner, state_vector, i)
            rewardLooser = ais[1].loose()
            ais[1].callbackGameStateChange(rewardLooser, state_vector, i)
            print(ais[2].name, " wins", ais[2].nbWin)
        else:
            rewardLooser1 = ais[1].draw()
            ais[1].callbackGameStateChange(rewardLooser1, state_vector, i)
            rewardLooser2 = ais[2].draw()
            ais[2].callbackGameStateChange(rewardLooser2, state_vector, i)
            print("Draw !")

    # ais[1] = AI("IA", 1, AITypes.ANN)
    # ais[2] = AI("IA", 1, AITypes.ANN, 20000)
    # DISPLAY_INFO = True
        #if i > 500:
            #plt.scatter(i, ais[2].nbWin)
            #plt.draw()
            #plt.pause(0.01)
    #plt.show()
    # Log results after we're done all frames.
