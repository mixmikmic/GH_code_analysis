import pandas as pd
import numpy as np

german_party = pd.read_csv('database/german-election-2017/2017_german_election_party.csv')
german_overall = pd.read_csv('database/german-election-2017/2017_german_election_overall.csv')

german_overall.groupby('state').sum()[['valid_first_votes', 'registered.voters']]

german_party.groupby(['state', 'party']).sum()['votes_first_vote']

most_votes = german_party.groupby(['state', 'party'])['votes_first_vote'].max()
most_votes.head()

best_areas = {}
for row in most_votes.iteritems():
    data = german_party[(german_party['state'] == row[0][0]) &
             (german_party['party'] == row[0][1]) &
             (german_party['votes_first_vote'] == row[1])].head(1)
    best_areas[(row[0][0], row[0][1])] = data['area_name'].iloc[0]
best_areas

least_index = german_party.groupby(['state', 'party']).apply(lambda group:group['votes_first_vote'].argmin())
german_party.loc[least_index][['state', 'party', 'area_name', 'votes_first_vote']]

registered = german_overall.groupby(['state', 'area_names']).sum()['registered.voters'].to_dict()
registered

german_party['registered_voters'] = german_party.apply(lambda row: registered[(row['state'], row['area_name'])], axis=1)
german_party.head()

german_party['percentage'] = german_party['votes_first_vote'] / german_party['registered_voters'] * 100
german_party.head()

least_perc_index = german_party.groupby(['state', 'party']).apply(lambda group:group['percentage'].argmin())
german_party.loc[least_perc_index][['state', 'party', 'area_name', 'percentage']]

max_perc_index = german_party.groupby(['state', 'party']).apply(lambda group:group['percentage'].argmax())
german_party.loc[max_perc_index][['state', 'party', 'area_name', 'percentage']]

german_party['difference'] = german_party['votes_first_vote'] - german_party['votes_second_vote']
german_party.groupby('area_name').sum()['difference']

german_party.groupby('state').sum()['difference']

german_party.groupby('party').sum()['difference']

german_party.groupby(['area_name', 'party']).sum()['difference']

