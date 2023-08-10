import SnakesLadders as SL
from pprint import PrettyPrinter
import pandas as pd
get_ipython().magic('matplotlib inline')

pp = PrettyPrinter(16)  # 16 is the column width of the output

game = SL.GameFSM(16)

# Make ladders
game.all_states[2].link = 10
game.all_states[8].link = 14

# Make snakes
game.all_states[11].link = 4
game.all_states[15].links = 6

game.make_state_kinds()

game.run()
pp.pprint(game.records)
print(SL.count_moves(game.records))
print(SL.count_snakes_and_ladders(game.records))

tot_moves = 0
all_moves = []
num_runs = 10000
for i in range(num_runs):
    game.run()
    moves = SL.count_moves(game.records)
    tot_moves += moves
    all_moves.append(moves)
print(tot_moves/num_runs)

max(all_moves)

df = pd.DataFrame({'moves': all_moves})

df.describe()



type(df['moves'].value_counts())

move_counts = df['moves'].value_counts()

df.hist(bins=max(move_counts.keys())-min(move_counts.keys()))

