import sgf
import pandas as pd
from go_utils import *
from matplotlib import pyplot as plt
import matplotlib
from AlphaGo.go import GameState
import numpy as np
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

with open("data/sample/2004-01-01-1.sgf") as f:
    collection = sgf.parse(f.read())

len(collection)

gt = collection[0]

collection[0].root.properties['AB']

for i in collection[0].rest:
    print(i.properties)
    break

node0 = gt.nodes[0]

dframe = pd.read_csv('data/sample/2004-01-01-1.sgf',header=None)

len(dframe)

gt = GameState()

#from AlphaGo.preprocessing.preprocessing import Preprocess
from AlphaGo.preprocessing.game_converter import GameConverter

feature_list = [
            "board",
            "ones",
            "turns_since",
            "liberties",
            "capture_size",
            "self_atari_size",
            "liberties_after",
            "ladder_capture",
            "ladder_escape",
            "sensibleness",
            "zeros"]
gc = GameConverter(feature_list)

game_iter = gc.convert_game('data/sample/2004-01-01-1.sgf',bd_size=19)
num = 4

    

i,j = game_iter.__next__()

np.transpose(i,[0,2,3,1]).shape,j

c = i[0][0] - i[0][1]

c.shape

print(j)
plot_board(c,figsize=(5,5),next_step=j)

def cod2array(cod,size=19):
    arr = np.zeros((size,size))
    arr[cod[0]][cod[1]] = 1
    return arr

print(j)
plot_board(cod2array(j))

from AlphaGo.util import sgf_to_gamestate
gs = sgf_to_gamestate(open('data/sample/2004-01-01-1.sgf').read())

plot_board(gs.board,figsize=(5,5))

gs.get_winner()

gs.board.shape

import os

go_prefix = 'data/chess-kgs/'
go_dirs = os.listdir(go_prefix)

sgfs = []
for one_dir in go_dirs:
    dirname = os.path.join(go_prefix,one_dir)
    dircontent = os.listdir(dirname)
    for one_sgf in dircontent:
        sgfs.append(str(os.path.join(go_prefix,one_dir,one_sgf)))

len(sgfs)

dframe = pd.DataFrame(sgfs)

dframe.to_csv('data/sgf_list.csv',header=None,index=None)

content = pd.read_csv('data/sgf_list.csv',header=None,index_col=None)

content = [i[0] for i in content.get_values()]

content[:10]

len(content)

import random

random.shuffle(content)

len(content)

gap = int(len(content) * 0.9)

train = content[:gap]
test = content[gap:]

trainframe = pd.DataFrame(train)
trainframe.to_csv('data/train_list.csv',header=None,index=None)

testframe = pd.DataFrame(test)
testframe.to_csv('data/test_list.csv',header=None,index=None)

len(trainframe),len(testframe)



