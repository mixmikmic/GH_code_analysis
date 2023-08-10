get_ipython().run_cell_magic('bash', '', '#clears all the pics\nrm -f pics/*')

get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')
get_ipython().run_line_magic('matplotlib', 'inline')
from core import Board
from random import choice
def test():
    rows, cols = 9, 6
    no_of_games = 1
    no_of_players = 3
    no_of_moves_avg = 0
    board = Board(rows=rows, cols=cols, no_of_players=no_of_players)
    cnt = 0
    while not board.game_complete():
        for i in range(no_of_players):
            pos = choice(board.valid_moves())
            board.play(pos, mid_states=False)
            #print("Move:" + str(pos))
            a = str(cnt)
            a = "0"*(4-len(a)) + a
            figname = "pics/fig" + a
            board.drawGraphical(figname)
            cnt += 1
            if board.game_complete():
                break
    assert(board.verify_game_over())
test()

get_ipython().run_cell_magic('bash', '', 'convert -delay 20 -loop 0 pics/*.png pics/myimage.gif')

from IPython.display import HTML
HTML('<img src="pics/myimage.gif">')

