from random import sample
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from pandas import DataFrame as df
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

class Dice:
    def __init__(self, checksum, alldice = False):
        self.checksum = checksum
        self.dice = dict()
        self.names = []
        self.data = []
        if alldice == True:
            self.add_all_players()
    def add_all_players(self):
        def create_name(die):
            return '.'.join([str(d) for d in die])
        
        levels = [[] for i in range(self.checksum+1)]
        levels[0].append([])
        for l in range(1,len(levels)):
            for j in range(1,l+1):
                for x in levels[l-j]:
                    if len(x) < 6 and (len(x) == 0 or x[-1] >= j):
                        levels[l].append([*x, j])
        for die in levels[-1]:
            die += (6-len(die))*[0]
            name = create_name(die)
            self.dice[name] = die
            self.names.append(name)
    def add_player(self, name, values):
        if len(values) == 6:
            if sum(values) == self.checksum:
                self.dice[name] = values
                self.dice[name].sort()
                self.names.append(name)
            else:
                print(name + ": Sum is incorrect!")
        else:
            print(name + ": A die has 6 sides!")
    def probability(self, name1, name2):
        w = 0.0
        l = 0.0
        d = 0.0
        for i in self.dice[name1]:
            for j in self.dice[name2]:
                if i > j:
                    w += 1
                elif i < j:
                    l += 1
                else:
                    d += 1
        return [round(x/36,3) for x in [w,l,d]]
    def allvsall(self):
        self.data = [[self.probability(name1,name2) for name1 in self.names] for name2 in self.names]
    def heatmap(self,f, cm):
        data = [[f(x) for x in a] for a in reversed(self.data)]
        plt.pcolor(data, cmap=cm)
        ax = plt.gca()
        plt.xticks([0.5+i for i in range(len(self.names))], self.names, rotation='vertical')
        plt.yticks([0.5+i for i in range(len(self.names))], list(reversed(self.names)), rotation='horizontal')
        ax.xaxis.set_tick_params(labeltop='on')
        plt.colorbar()
        plt.show()
    def compute_stats(self):
        self.prob_stats = df()
        self.match_stats = df()
        data = self.data[:][:][:]
        for i in range(len(data)):
            data[i] = data[i][:i] + data[i][i+1:]
        self.prob_stats["wins"] = [round(np.mean([x[1] for x in p]),3) for p in data]
        self.prob_stats["loses"] = [round(np.mean([x[0] for x in p]),3) for p in data]
        self.prob_stats["draws"] = [round(np.mean([x[2] for x in p]),3) for p in data]
        self.prob_stats["wins-loses"] = self.prob_stats["wins"] - self.prob_stats["loses"]
        self.match_stats["wins"] = [int(np.sum([x[1] > x[0] for x in p])) for p in data]
        self.match_stats["loses"] = [int(np.sum([x[0] > x[1] for x in p])) for p in data]
        self.match_stats["draws"] = [int(np.sum([x[0] == x[1] for x in p])) for p in data]
        self.match_stats["wins- loses"] = self.match_stats["wins"] - self.match_stats["loses"]
        self.match_stats["wins/loses"] = np.round(np.divide(self.match_stats["wins"], self.match_stats["loses"]), 3)
        self.prob_stats.index = self.names
        self.match_stats.index = self.names
    def compute_all(self):
        self.allvsall()
        self.compute_stats()

game = Dice(21, alldice=True)
game.compute_all()

game = Dice(21) 
game.add_player("Veronika1",[6,5,4,3,2,1])
game.add_player("Mišo1",[5,5,5,4,1,1])
game.add_player("Samko1",[5,5,3,3,4,1])
game.add_player("KUBO",[7,7,7,0,0,0])
game.add_player("Tonda",[1,4,4,4,4,4])
game.add_player("Hanka",[8,6,6,1,0,0])
game.add_player("Samko2",[10,10,1,0,0,0])
game.add_player("Andrej",[4,4,4,4,4,1])
game.add_player("Matúš",[10,5,5,1,0,0])
game.add_player("Dominik",[6,6,6,1,1,1])
game.add_player("Zvono",[5,5,5,6,0,0])

game.compute_all()

game.heatmap(lambda x: x[1], "Blues")

game.heatmap(lambda x: x[0], "Blues")

game.heatmap(lambda x: x[2], "Blues")

game.heatmap(lambda x: x[1] - x[0], "seismic")

game.heatmap(lambda x: np.sign(x[1] - x[0]), "seismic")

top = 20
game.prob_stats.sort_values("wins-loses", ascending=False)[:top]
#game.prob_stats.sort_values("wins", ascending=False)[:top]
#game.prob_stats.sort_values("loses")[:top]
#game.prob_stats.sort_values("draws", ascending=False)[:top]

top = 20
game.match_stats.sort_values("wins-loses", ascending=False)[:top]
#game.match_stats.sort_values("wins/loses", ascending=False)[:top]
#game.match_stats.sort_values("wins", ascending=False)[:top]
#game.match_stats.sort_values("loses")[:top]
#game.match_stats.sort_values("draws", ascending=False)[:top]

