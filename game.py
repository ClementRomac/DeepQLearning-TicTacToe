import gym
import gym_tictactoe

from AI import AI
from AI import AITypes
from AI import RewardTypes

def play_a_game(AIs, nb_game, gym_env):
    done = False
    nb_frames_per_game = 0
    player_index = 1
    # Reset the env before playing
    state = env.reset()

    while not done:
        player = AIs[player_index]
        index = player.play(state, nb_game)
        state, reward, done, infos = env.step(index, player.symbol)
        if DISPLAY_INFO:
            env.render(mode=None)

        if not done:
            player.callback_game_state_changed(player.get_reward(RewardTypes.NOTHING), state, nb_game)
            player_index = 1 + (player_index % 2)
            nb_frames_per_game += 1
        else:
            if reward == player.get_reward(RewardTypes.DRAW):
                if DISPLAY_INFO:
                    print("It's a draw !")
                return None, nb_frames_per_game, state
            elif reward == player.get_reward(RewardTypes.LOOSE):
                print("bad position")
                return 1 + (player_index % 2), nb_frames_per_game, state
            elif reward == player.get_reward(RewardTypes.WIN):
                if DISPLAY_INFO:
                    print("Winner: " + player.name)
                return player_index, nb_frames_per_game, state

def play_games(nbr, AIs, gym_env, checkpoint=0):
    for i in range(checkpoint, checkpoint + nbr + 1):
        winner_index, nb_play, state_vector = play_a_game(AIs, i, gym_env)
        if winner_index is not None:
            reward_winner = AIs[winner_index].get_reward(RewardTypes.WIN)
            AIs[winner_index].callback_game_state_changed(reward_winner, state_vector, i)
            rewardLooser = AIs[1 + (winner_index % 2)].get_reward(RewardTypes.LOOSE)
            AIs[1 + (winner_index % 2)].callback_game_state_changed(rewardLooser, state_vector, i)
            print("Game ", i, " : ", AIs[winner_index].name, " wins for the ", AIs[winner_index].nb_win, " times in ", nb_play + 1, " plays")
        else:
            rewardLooser1 = AIs[1].get_reward(RewardTypes.DRAW)
            AIs[1].callback_game_state_changed(rewardLooser1, state_vector, i)
            rewardLooser2 = AIs[2].get_reward(RewardTypes.DRAW)
            AIs[2].callback_game_state_changed(rewardLooser2, state_vector, i)
            print("Game ", i, " : ", "Draw !")

if __name__ == '__main__':

    ####################### TRAINING #######################

    # print("-------------------- TRAINGING VS RANDOM --------------------")
    #
    # AIs = {
    #     1: AI("AI", 1, AITypes.ANN, isTraining=True),
    #     2: AI("Random Player", -1, AITypes.RANDOM)
    # }
    #
    # DISPLAY_INFO = False
    #
    # playGames(80000, AIs)

    # print("-------------------- TRAINGING VS ITSELF --------------------")
    #
    # trained_ai = AI.load_ai("2017.9.23/80001")
    # AIs = {
    #     1: trained_ai,
    #     2: trained_ai
    # }
    #
    # DISPLAY_INFO = False
    #
    # playGames(20000, AIs, checkpoint=80000)



    ####################### PLAYING #######################

    print("-------------------- IA VS HUMAN --------------------")

    AIs = {
        1: AI("AI", 1, AITypes.ANN, load_weights="2017.9.23/57000"),
        2: AI("Human Player", -1, AITypes.HUMAN)
    }

    global DISPLAY_INFO
    DISPLAY_INFO = True
    env = gym.make('TicTacToe-v1')
    env.init([ai.symbol for _, ai in AIs.items()])

    play_games(10000, AIs, env)