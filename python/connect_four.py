from aima3.games import (ConnectFour, RandomPlayer, QueryPlayer, players,
                         MCTSPlayer, MiniMaxPlayer, AlphaBetaCutoffPlayer,
                         AlphaBetaPlayer)

p1 = RandomPlayer("Rando")
p2 = AlphaBetaCutoffPlayer("Alphie")
game = ConnectFour()
game.play_game(p1, p2)

game.display(game.initial)

p1.get_action(game.initial, turn=1)

p2.get_action(game.initial, turn=1)

state = game.initial
turn = 1
for i in range(3):
    print("Current state:")
    game.display(state)
    action = p1.get_action(state, round(turn))
    state = game.result(state, action)
    print("Made the action: %s" % (action, ))
    turn += .5
game.display(state)

for i,p in enumerate(players):
    print(i, p.name)

game.play_matches(10, players[4], players[3])

game.play_tournament(10, players[4], players[3], players[0])

