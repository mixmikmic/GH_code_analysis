import matplotlib
import pandas as pd

pd.options.display.max_columns = 50

get_ipython().magic('matplotlib inline')

ideas = pd.read_csv('wikisurvey_2565_ideas_2016-12-14T20-42-29Z.csv')
votes = pd.read_csv('wikisurvey_2565_votes_2016-12-14T20-42-35Z.csv')
nonvotes = pd.read_csv('wikisurvey_2565_nonvotes_2016-12-14T20-42-38Z.csv')

ideas.head(1)

len(ideas)

len(votes)

votes.head(1)

len(votes['Session ID'].unique())

len(votes['Hashed IP Address'].unique())

len(votes['Winner ID'].unique())

len(votes['Loser ID'].unique())

votes.groupby('Hashed IP Address').count()['Vote ID']

votes.groupby('Session ID').count()['Vote ID']

votes.groupby('Session ID').count()['Vote ID'].hist(bins=50)

data = votes[['Winner ID', 'Loser ID', 'Hashed IP Address']].copy()
data.head()

data['value'] = (data['Winner ID'] > data['Loser ID']).apply(int)
data.head()

data['bigIndex'] = data[['Winner ID', 'Loser ID']].apply(max, axis=1)
data['smallIndex'] = data[['Winner ID', 'Loser ID']].apply(min, axis=1)
data.head()

data['userId'] = data['Hashed IP Address']

# data['smallIndex'] -= data['smallIndex'].min()
# data['bigIndex'] -= data['bigIndex'].min()

item_map = {}
for i, item_id in enumerate(ideas['Idea ID'].sort_values()): 
    item_map[item_id] = i
    
text_map = {}
id_to_text = ideas.set_index('Idea ID')['Idea Text']
for i, item_id in enumerate(ideas['Idea ID']):
    text_map[i] = id_to_text[item_id]

pd.DataFrame.from_dict(item_map, orient='index').to_csv('2565_item_map.csv', header=False)

pd.DataFrame.from_dict(text_map, orient='index').to_csv('2565_text_map.csv', header=False)

for item in item_map:
    data.loc[data['smallIndex'] == item, 'smallIndex'] = item_map[item]
    data.loc[data['bigIndex'] == item, 'bigIndex'] = item_map[item]

data.head()

data[['smallIndex', 'bigIndex', 'value', 'userId']].to_csv('2565_dat.csv', index=False, header=False)
pd.read_csv('2565_dat.csv', header=None)

