### Import
import GetPbP
import PbPMethods2 as pm2
from pylab import *
get_ipython().magic('matplotlib inline')
import seaborn

### Data setup
gd = {}
gp = {}
for season in range(2007, 2016):
    for team in pm2.get_teams(season):
        seasonteam = '{0:d}{1:s}'.format(season, team)
        gd[seasonteam] = 0
        gp[seasonteam] = set()
        for line in pm2.read_team_pbp(team, season, strengths='all', types=['GOAL']):
            g = pm2.get_game(line)
            if g > 21230:
                break
            else:
                gp[seasonteam].add(g)
                if pm2.get_acting_team(line) == team:
                    gd[seasonteam] += 1
                else:
                    gd[seasonteam] -= 1
    print('Done with', season)
gp2 = {key: len(val) for key, val in gp.items()}
gdpergm = {key: gd[key]/gp2[key] for key in gp}

for key, val in gdpergm.items():
    if val > 0.6:
        print(key, val)

standings = {'2007DET': 115, 
             '2008DET': 112,
             '2008BOS': 116,
             '2008S.J': 117,
             '2009WSH': 121,
             '2009VAN': 103,
             '2009CHI': 112,
             '2010BOS': 103,
             '2010VAN': 117,
             '2011PIT': 108,
             '2011BOS': 102,
             '2012CHI': 77/48*82,
             '2012PIT': 72/48*82,
             '2013ANA': 116,
             '2013BOS': 117,
             '2013STL': 111,
             '2013S.J': 111,
             '2014NYR': 113,
             '2015WSH': 120}

x = [gdpergm[key] for key in standings]
y = [standings[key] for key in standings]
scatter(x, y, c='lightgreen', s=200, alpha=0.5)
for key in standings:
    annotate(key, xy=(gdpergm[key], standings[key]), ha='center', va='center')
xlabel('GD per game')
ylabel('Standings points per 82')
title('Goal differential vs standings, 2007-16, top GD teams')    
from scipy.stats import linregress
m, b, r, p, e = linregress(x, y)
xmin, xmax = xlim()
ymin, ymax = ylim()
plot([xmin, xmax], [ymin, ymax], lw=3)
xlim(xmin, xmax)
ylim(ymin, ymax)



