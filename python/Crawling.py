get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import requests
import networkx as nx
from bs4 import BeautifulSoup, NavigableString

raw = requests.get('http://www.sports-reference.com/cbb/postseason/2017-ncaa.html').text
soup = BeautifulSoup(raw,'lxml')

east_soup = soup.find_all('div',{'id':'east'})[0]
midwest_soup = soup.find_all('div',{'id':'midwest'})[0]
south_soup = soup.find_all('div',{'id':'south'})[0]
west_soup = soup.find_all('div',{'id':'west'})[0]

def get_teams(division_soup):
    teams_href_list = list()
    first_round_teams = list(division_soup.find_all('div',{'class':'round'})[0].children)
    for team_soup in first_round_teams:
        if type(team_soup) != NavigableString:
            pairs = team_soup.find_all('a')
            for team in pairs:
                if team.text != 'tbd':
                    teams_href_list.append(team)
    return teams_href_list

east_teams_2017 = get_teams(east_soup)
midwest_teams_2017 = get_teams(midwest_soup)
south_teams_2017 = get_teams(south_soup)
west_teams_2017 = get_teams(west_soup)
teams_2017 = east_teams_2017 + midwest_teams_2017 + south_teams_2017 + west_teams_2017
teams_2017

for team in teams_2017:
    name = "2017-{0}".format(team.text)
    url = 'http://www.sports-reference.com/' + team['href'].replace('2017.html','2017-schedule.html')
    team_raw = requests.get(url).text
    team_soup = BeautifulSoup(team_raw,'lxml')
    df = pd.read_html(str(team_soup.find_all('table',{'class':'sortable stats_table','id':'schedule'})[0]))[0]
    df = df.set_index('G')
    df.to_csv(name + '.csv')

all_results = list()
for team in teams_2017:
    url = 'http://www.sports-reference.com/' + team['href'].replace('2017.html','2017-schedule.html')
    team_raw = requests.get(url).text
    team_soup = BeautifulSoup(team_raw,'lxml')
    table = team_soup.find_all('table',{'class':'sortable stats_table','id':'schedule'})[0]
    date = [i['csk'] for i in table.find_all('td',{'data-stat':'date_game'})]
    opponents = [i.text.split('\xa0')[0] for i in table.find_all('td',{'data-stat':'opp_name'})]
    results = [i.text for i in table.find_all('td',{'data-stat':'game_result'})]
    team_score = [i.text for i in table.find_all('td',{'data-stat':'pts'})]
    opponent_score = [i.text for i in table.find_all('td',{'data-stat':'opp_pts'})]
    team_results = list(zip([team.text]*len(opponents),opponents,date,results,team_score,opponent_score))[:-1]
    for team_result in team_results:
        all_results.append(team_result)

[i for i in all_results if i[0] == 'Louisville']

pd.DataFrame(all_results,columns=['team','opponent','date','result','score','opp_score']).to_csv('results.csv')

reduced_results = list()
for (team, opponent, date, result, score, opp_score) in all_results:
    if score > opp_score:
        reduced_results.append((team, opponent, date, score, opp_score))
        
len(all_results), len(reduced_results)

tournament_teams = [i.text for i in teams_2017]

g = nx.DiGraph()

for (team, opponent, date, score, opp_score) in reduced_results:
    if opponent in tournament_teams:
        differential = int(score) - int(opp_score)
        if g.has_edge(team,opponent):
            g[team][opponent]['weight'] += differential
        else:
            g.add_edge(team, opponent, weight = abs(differential))
        
print("There are {0} nodes and {1} edges in the network".format(g.number_of_nodes(), g.number_of_edges()))

nx.write_gexf(g,'tournament_schedule.gexf')

d = {}
for node in g.nodes():
    d[node] = np.sum([g[node][successor]['weight'] for successor in g.successors(node)])
    
pd.Series(d).sort_values(ascending=False)

pd.Series(nx.neighbor_degree.average_neighbor_degree(g,source='out',target='out',weight='weight')).sort_values(ascending=False)



