import pandas as pd

german_party = pd.read_csv('../database/german-election-2017/2017_german_election_party.csv')

german_party.head()

german_overall = pd.read_csv('../database/german-election-2017/2017_german_election_overall.csv')

german_overall.head()

german_overall['perc'] = german_overall['total_votes'] / german_overall['registered.voters'] * 100

german_overall.head()

states = list(set(german_overall['state']))
states

votes = {}
for state in states:
    votes[state] = sum(german_overall[german_overall['state'] == state]['total_votes'])
votes

registered = {}
for state in states:
    registered[state] = sum(german_overall[german_overall['state'] == state]['registered.voters'])
registered

parties = list(set(german_party['party']))
parties

import collections
v = {}
for state in states:
    v[state] = collections.Counter()
    single_state = german_party[german_party.state == state]
    for party in list(set(single_state['party'])):
        v[state][party] = sum(single_state[single_state.party == party]['votes_first_vote'])
v

for state in states:
    print(state, v[state].most_common(1))



