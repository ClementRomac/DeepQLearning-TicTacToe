from IPython.core.display import display
import matplotlib.pyplot as plt

from AI import AI
from AI import AITypes
from AI import RewardTypes

DISPLAY_INFO = False


# ------------------------------------------ DISPLAY ----------------------------------------

def displayGrid(grid):
    printGridLine(grid)
    printGridLine(grid, 3)
    printGridLine(grid, 6)
    print(" -------------")

    print()


def printGridLine(grid, offset=0):
    print(" -------------")
    for i in range(3):
        if grid[i + offset] == 0:
            print(" | " + " ", end='')
        else:
            print(" | " + str(grid[i + offset]), end='')
    print(" |")


# ------------------------------------------ GAME STATE CHECK ----------------------------------------
def isWin(grid):
    if (grid[0] == grid[1]) and (grid[0] == grid[2]) and (grid[0] != 0):
        return True
    elif (grid[3] == grid[4]) and (grid[3] == grid[5]) and (grid[3] != 0):
        return True
    elif (grid[6] == grid[7]) and (grid[6] == grid[8]) and (grid[6] != 0):
        return True
    elif (grid[0] == grid[3]) and (grid[0] == grid[6]) and (grid[0] != 0):
        return True
    elif (grid[1] == grid[4]) and (grid[1] == grid[7]) and (grid[1] != 0):
        return True
    elif (grid[2] == grid[5]) and (grid[2] == grid[8]) and (grid[2] != 0):
        return True
    elif (grid[0] == grid[4]) and (grid[0] == grid[8]) and (grid[0] != 0):
        return True
    elif (grid[2] == grid[4]) and (grid[2] == grid[6]) and (grid[2] != 0):
        return True
    else:
        return False


def isDraw(grid):
    for i in range(9):
        if grid[i] == 0:
            return 0
    return 1


# ------------------------------------------ ACTIONS ----------------------------------------
def getPlayablePosition(state_vector_param):
    playable_position = []
    i = 0
    while i < len(state_vector_param):
        if state_vector_param[i] == 0:
            playable_position.append(i)
        i += 1
    return playable_position


def stroke(state_vector_param, joueur, nbGame):
    positionIsBad = False
    playable_position = getPlayablePosition(state_vector_param)

    index = joueur.play(playable_position, state_vector_param, nbGame)

    if state_vector_param[index] != 0:
        positionIsBad = True

    state_vector_param[index] = joueur.symbol

    if DISPLAY_INFO:
        print(joueur.name, " (", joueur.symbol, ")", sep='')
        displayGrid(state_vector_param)

    return positionIsBad


# ------------------------------------------ GAME ----------------------------------------
def deepCopy(state_vector):
    newVector = []
    for i in state_vector:
        newVector.append(i)
    return newVector


def playAGame(AIs, nbGame):
    winner = False
    nb_frames_per_game = 0
    player, state_vector = init()
    while not winner:
        stateVectorLooser = deepCopy(state_vector)
        if stroke(state_vector, AIs[player], nbGame):
            print("bad position")
            return 1 + (player % 2), nb_frames_per_game, state_vector, stateVectorLooser

        if isWin(state_vector):
            if DISPLAY_INFO:
                print("Winner: " + AIs[player].name)
            return player, nb_frames_per_game, state_vector, stateVectorLooser
        elif isDraw(state_vector):
            if DISPLAY_INFO:
                print("It's a draw !")
            return None, nb_frames_per_game, state_vector, stateVectorLooser

        AIs[player].callbackGameStateChange(AIs[player].getReward(RewardTypes.NOTHING), state_vector, nbGame)

        player = 1 + (player % 2)
        nb_frames_per_game += 1


def playGames(nbr, AIs):
    plt.xlabel('Nb of Games')
    plt.ylabel('Human Wins')
    plot = plt.plot(0, 0)

    for i in range(0, nbr + 1):
        winnerIndex, nbPlay, stateVectorWinner, stateVectorLooser = playAGame(AIs, i)
        if winnerIndex is not None:
            if winnerIndex == 2 and i > 2500:
                displayGrid(stateVectorWinner)
                print(stateVectorLooser)
                print(stateVectorWinner)
            rewardWinner = AIs[winnerIndex].getReward(RewardTypes.WIN)
            AIs[winnerIndex].callbackGameStateChange(rewardWinner, stateVectorWinner, i)
            rewardLooser = AIs[1 + (winnerIndex % 2)].getReward(RewardTypes.LOOSE)
            AIs[1 + (winnerIndex % 2)].callbackGameStateChange(rewardLooser, stateVectorLooser, i)
            print(AIs[winnerIndex].name, " wins for the ", AIs[winnerIndex].nbWin, " times in ", nbPlay, " plays")
        else:
            rewardLooser1 = AIs[1].getReward(RewardTypes.DRAW)
            AIs[1].callbackGameStateChange(rewardLooser1, stateVectorWinner, i)
            rewardLooser2 = AIs[2].getReward(RewardTypes.DRAW)
            AIs[2].callbackGameStateChange(rewardLooser2, stateVectorWinner, i)
            print("Draw !")
        if i % 10 == 0:
            plt.scatter(i, AIs[1].nbWin, color='r')
            plt.scatter(i, AIs[2].nbWin)
            plt.draw()
            plt.pause(0.01)
        # plt.show()


def init():
    player = 1
    state_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    return player, state_vector


if __name__ == '__main__':
    AIs = {
        1: AI("AI", 1, AITypes.ANN),
        2: AI("Random Player", 2, AITypes.RANDOM)
    }

    print("-------------------- TRAINGING VS RANDOM --------------------")
    playGames(20000, AIs)

    # AIs[1] = AI("IA", 1, AITypes.ANN)
    # AIs[2] = AI("IA", 2, AITypes.ANN, 20000)
    # print("-------------------- TRAINGING VS ITSELF --------------------")
    # playGames(10000, AIs)
