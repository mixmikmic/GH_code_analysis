from games import (GameState, Game, Fig52Game, TicTacToe, query_player, random_player, 
                    alphabeta_player, play_game, minimax_decision, alphabeta_full_search,
                    alphabeta_search, Canvas_TicTacToe)

get_ipython().run_line_magic('psource', 'Game')

get_ipython().run_line_magic('psource', 'TicTacToe')

game52 = Fig52Game()

print(random_player(game52, 'A'))
print(random_player(game52, 'A'))

print( alphabeta_player(game52, 'A') )
print( alphabeta_player(game52, 'B') )
print( alphabeta_player(game52, 'C') )

minimax_decision('A', game52)

alphabeta_full_search('A', game52)

play_game(game52, alphabeta_player, alphabeta_player)

play_game(game52, alphabeta_player, random_player)

#play_game(game52, query_player, alphabeta_player)
#play_game(game52, alphabeta_player, query_player)

ttt = TicTacToe()

ttt.display(ttt.initial)

my_state = GameState(
    to_move = 'X',
    utility = '0',
    board = {(1,1): 'X', (1,2): 'O', (1,3): 'X',
             (2,1): 'O',             (2,3): 'O',
             (3,1): 'X',
            },
    moves = [(2,2), (3,2), (3,3)]
    )

ttt.display(my_state)

random_player(ttt, my_state)

random_player(ttt, my_state)

alphabeta_player(ttt, my_state)

print(play_game(ttt, random_player, alphabeta_player))

for _ in range(10):
    print(play_game(ttt, alphabeta_player, alphabeta_player))

for _ in range(10):
    print(play_game(ttt, random_player, alphabeta_player))

bot_play = Canvas_TicTacToe('bot_play', 'random', 'alphabeta')

rand_play = Canvas_TicTacToe('rand_play', 'human', 'random')

ab_play = Canvas_TicTacToe('ab_play', 'human', 'alphabeta')

