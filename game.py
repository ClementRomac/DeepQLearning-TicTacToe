from IPython.core.display import display

from IA import IA
import matplotlib.pyplot as plt

DISPLAY_INFO = True
REWARD_ONE_HIT = 1


def displayGrid(grille):
    print_line(grille)
    print_line(grille, 3)
    print_line(grille, 6)
    print(" -------------")

    print()


def print_line(grille, offset=0):
    print(" -------------")
    for i in range(3):
        if grille[i + offset] == 0:
            print(" | " + " ", end='')
        else:
            print(" | " + str(grille[i + offset]), end='')
    print(" |")


def est_gagnant(grille):
    if (grille[0] == grille[1]) and (grille[0] == grille[2]) and (grille[0] != 0):
        return 1
    if (grille[3] == grille[4]) and (grille[3] == grille[5]) and (grille[3] != 0):
        return 1
    if (grille[6] == grille[7]) and (grille[6] == grille[8]) and (grille[6] != 0):
        return 1
    if (grille[0] == grille[3]) and (grille[0] == grille[6]) and (grille[0] != 0):
        return 1
    if (grille[1] == grille[4]) and (grille[1] == grille[7]) and (grille[1] != 0):
        return 1
    if (grille[2] == grille[5]) and (grille[2] == grille[8]) and (grille[2] != 0):
        return 1
    if (grille[0] == grille[4]) and (grille[0] == grille[8]) and (grille[0] != 0):
        return 1
    if (grille[2] == grille[4]) and (grille[2] == grille[6]) and (grille[2] != 0):
        return 1


def tour(state_vector_param, joueur, nbGame):
    playable_position = []
    positionIsBad = False
    i = 0
    while i < len(state_vector_param):
        if state_vector_param[i] == 0:
            playable_position.append(i)
        i += 1
    index = joueur.play(playable_position, state_vector_param, nbGame)

    if state_vector_param[index] != 0:
        # raise Exception("index not empty :(", state_vector, index)
        positionIsBad = True

    state_vector_param[index] = joueur.symbol

    if DISPLAY_INFO:
        print(joueur.name, " (", joueur.symbol, ")", sep='')
        displayGrid(state_vector_param)

    return positionIsBad


def est_match_nul(grille):
    for i in range(9):
        if grille[i] == 0:
            return 0
    return 1


def playAGame(player, nbGame):
    winner = False
    nb_frames_per_game = 0
    while not winner:
        positionIsBad = tour(state_vector, ias[player], nbGame)
        if positionIsBad:
            print("bad position")
            if player == 1:
                return ias[2], nb_frames_per_game, state_vector
            elif player == 2:
                return ias[1], nb_frames_per_game, state_vector

        if est_gagnant(state_vector):
            if DISPLAY_INFO:
                print("Gagnant: " + ias[player].name)
            return ias[player], nb_frames_per_game, state_vector
        else:
            if est_match_nul(state_vector):
                if DISPLAY_INFO:
                    print("Plus de place ! Match nul !")
                return None, nb_frames_per_game, state_vector
        ias[player].callbackGameStateChange(REWARD_ONE_HIT, state_vector, nbGame)
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
    global ias
    ias = {1: IA("IA", 1), 2: IA("Human", 2, isDummy=True, isHuman=DISPLAY_INFO)}
    #plt.xlabel('Nb of Games')
    #plt.ylabel('Human Wins')
    #plot = plt.plot(0,0)
    #plt.axis((500, 10000, 0, 50))


    for i in range(0, 20001):
        player, state_vector = init()
        winner, nbPlay, state_vector = playAGame(player, i)
        if winner == ias[1]:
            rewardWinner = ias[1].win(nbPlay)
            ias[1].callbackGameStateChange(rewardWinner, state_vector, i)
            rewardLooser = ias[2].loose()
            ias[2].callbackGameStateChange(rewardLooser, state_vector, i)
            print(ias[1].name, " win", ias[1].nbWin)
        elif winner == ias[2]:
            rewardWinner = ias[2].win(nbPlay)
            ias[2].callbackGameStateChange(rewardWinner, state_vector, i)
            rewardLooser = ias[1].loose()
            ias[1].callbackGameStateChange(rewardLooser, state_vector, i)
            print(ias[2].name, " win", ias[2].nbWin)
        else:
            rewardLooser1 = ias[1].equal()
            ias[1].callbackGameStateChange(rewardLooser1, state_vector, i)
            rewardLooser2 = ias[2].equal()
            ias[2].callbackGameStateChange(rewardLooser2, state_vector, i)
            print("No one win")
        #if i > 500:
            #plt.scatter(i, ias[2].nbWin)
            #plt.draw()
            #plt.pause(0.01)
    #plt.show()
    # Log results after we're done all frames.
