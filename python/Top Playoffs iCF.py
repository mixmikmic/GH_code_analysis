import PbPMethods2 as pm2
import pandas as pd

dfs = []
for season in range(2007, 2017):
    dfs.append(pd.read_csv(pm2.get_gamebygame_data_filename(season)))
dfs = pd.concat(dfs)
dfs.head()

playoffs = dfs.query('Game > 30000')
playoffs = playoffs[['Player', 'Team', 'Game', 'Season', 'TOION(60s)', 'iCF']]
playoffs['Round'] = (playoffs['Game'] - 30000) // 100
playoffs.head()

player_gp = playoffs.groupby(['Player', 'Team', 'Season', 'Round']).count()
player_gp.reset_index(inplace = True)
player_gp.rename(columns = {'Game': 'GP'}, inplace = True)
player_gp = player_gp[['Player', 'Team', 'Season', 'Round', 'GP']]
player_gp.head()

player_rounds = playoffs.groupby(['Player', 'Team', 'Season', 'Round']).sum()
player_rounds.drop(['Game'], axis = 1, inplace = True)
player_rounds.reset_index(inplace = True)

#Join to get the gp column
player_rounds = player_rounds.merge(player_gp, on = ['Player', 'Team', 'Season', 'Round'], how = 'inner')

player_rounds['iCF60'] = player_rounds['iCF'] / player_rounds['TOION(60s)']
player_rounds.sort_values(by = 'iCF60', ascending = False, inplace = True)
player_rounds.head()

temp = player_rounds.query('GP >= 6')
temp.reset_index(inplace = True, drop = True) #so index = ranks
temp.head(20)

