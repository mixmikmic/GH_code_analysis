import pandas as pd
import numpy as np

early_access = pd.read_csv('../data/early_access.csv')
early_access.head()
#pd.read_csv(file, index_col=0, parse_dates=True)

games = pd.read_csv('../data/games.csv')
games.head()

early_access.columns = ['Rank', 'name', 'Release date', 'Price', 'Score rank(Userscore/Metascore)', 'Owners', 'Playtime(Median)', 'Developer(s)', 'Publisher(s)']
early_access.head()

early_access_merged = pd.merge(games, early_access, on='name', how='inner')
early_access_merged.head()

ex_early_access = pd.read_csv('../data/ex_early_access.csv')
ex_early_access.head()

ex_early_access.columns = ['Rank', 'name', 'Release date', 'Price', 'Score rank(Userscore/Metascore)', 'Owners', 'Players', 'Playtime(Median)', 'Developer(s)', 'Publisher(s)']
ex_early_access.head()

ex_early_access_merged = pd.merge(games, ex_early_access, on='name', how='inner')
ex_early_access_merged.head()

early_access_merged['is early access?'] = True
ex_early_access_merged['is early access?'] = False

full_table = pd.merge(early_access_merged, ex_early_access_merged, how='outer')
full_table.to_pickle('../data/early_access pickle')
full_table.head()

# Changed NaN in release date for early access to not released because the game is not released yet
full_table['Release date'].fillna('Not released', inplace=True)
full_table.head()

#In the Price column, the $ sign was removed and made Free into 0 so that it would be easier to manipulate data
ex_early_access_merged['Price'] = ex_early_access_merged['Price'].str.replace('$', '')
ex_early_access_merged['Price'] = ex_early_access_merged['Price'].str.replace('Free', '0')

full_table['Price'] = full_table['Price'].str.replace('$', '')
full_table['Price'] = full_table['Price'].str.replace('Free', '0')

#Converting all values in Price column into numeric values 
ex_early_access_merged['Price'] = pd.to_numeric(ex_early_access_merged['Price'], errors='coerce')
print ex_early_access_merged['Price'].head()

full_table['Price'] = pd.to_numeric(ex_early_access_merged['Price'], errors='coerce')
print full_table['Price'].head()

#ex_early_access_merged

#average price per game 
print ex_early_access_merged['Price'].mean()

print full_table['Price'].mean()

ex_early_access_merged['Rank'].replace('',np.NaN) #replaces blank with NaN
ex_early_access_merged['Rank'].max() #highest rank seems to be 1043

full_table['Rank'].replace('',np.NaN) #replaces blank with NaN
full_table['Rank'].max() #highest rank seems to be 1574

ex_early_access_merged['Rank'].median() #checking to make sure it gives back a number

# seems the most expensive game is $65, I would think there would be a game more expensive than 65 bucks.
ex_early_access_merged['Price'].max() 

full_table['Price'].max() 

import matplotlib.pyplot as plt

# This graph shows distribution of price of games in our data set 
bins= [0,1,5,10,20,30,40,50,65]
plt.hist(ex_early_access_merged.Price, bins=bins, color="y")
plt.xticks(bins)
plt.show() 

ex_early_access_merged['year'] = ex_early_access_merged['Release date'].str.split(',').str[1]
ex_early_access_merged.head()

full_table['year'] = full_table['Release date'].str.split(',').str[1]
full_table['year'].fillna('Not released', inplace=True)
full_table.head()

full_table.to_csv('../data/full_table')

dff2 = ex_early_access_merged[['Score rank(Userscore/Metascore)', 'year']]
dff2.head()

#Converting all values in year column into numeric values 
dff2['year'] = pd.to_numeric(dff2['year'], errors='coerce')
dff2.head()

dff2['Score (%)'] = dff2['Score rank(Userscore/Metascore)'].str.split('%').str[0]
dff2['Score (%)'] = dff2['Score rank(Userscore/Metascore)'].str.split('(').str[0]
dff3 = dff2[['Score (%)','year']]
dff3['Score (%)'] = dff3['Score (%)'].str.replace('N/A', '0')
dff3['Score (%)'] = dff3['Score (%)'].str.replace('%', '')
dff3['Score (%)'] = pd.to_numeric(dff3['Score (%)'], errors='coerce')
dff3[(dff3 != 0).all(1)].head()



