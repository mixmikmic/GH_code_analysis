import numpy as np
import pandas as pd

game_df = pd.read_csv('2013-2016_NCAA_football_fcs_elo.csv')

game_df.head()

game_df['scorediff'] = (game_df['teamscore']-game_df['oppscore'])

game_df.head()

elo_ratings_dict = {}
for index, row in game_df.iterrows():
    elo_ratings_dict[row['team']] = 1500
    # also go through opponent column in case any team only in that column
    elo_ratings_dict[row['opponent']] = 1500

# function to update elo ratings based on the score
def elo_update(winner_elo, loser_elo, mov):
    k = 20
    mov_multiplier = (2.2 * np.log(mov + 1))/((0.001 * (winner_elo-loser_elo))+2.2)
    
    u_winner = 1/(1+(10 ** ((winner_elo-loser_elo)/400)))
    u_loser = 1/(1+(10 ** ((loser_elo-winner_elo)/400)))
    
    s1 = 1
    s2 = 0
    
    new_winner_elo = winner_elo + k * mov_multiplier * (s1 - u_winner)
    new_loser_elo = loser_elo + k * mov_multiplier * (s2 - u_loser)
    
    return [new_winner_elo, new_loser_elo]

current_year = 2013
for index, row in game_df.iterrows():
    # first check if it is a new year
    year = row['year']
    if year != current_year:
        # if new year, regress all elo ratings to the mean
        for key in elo_ratings_dict:
            elo_ratings_dict[key] = (0.5*elo_ratings_dict[key]) + (0.5*1500)
        # update the year
        current_year = year
        
    # find the margin of victory
    score_diff = row['scorediff']
    mov = abs(score_diff)
        
    # check if the game was at home, away, or neutral
    if row['location'] == 1.0:
        home_team = row['team']
        away_team = row['opponent']
        
        # get corresponding elo ratings
        home_elo = elo_ratings_dict[home_team] + 65
        away_elo = elo_ratings_dict[away_team]
        
        # check if home or away team won
        if score_diff > 0: 
            new_home_elo = elo_update(home_elo,away_elo,mov)[0] - 65
            new_away_elo = elo_update(home_elo,away_elo,mov)[1]
        elif score_diff < 0:
            new_away_elo = elo_update(away_elo,home_elo,mov)[0]
            new_home_elo = elo_update(away_elo,home_elo,mov)[1] - 65 
            
        # update the ratings dict
        elo_ratings_dict[home_team] = new_home_elo
        elo_ratings_dict[away_team] = new_away_elo
        
    elif row['location'] == -1.0:
        home_team = row['opponent']
        away_team = row['team']
        
        # get corresponding elo ratings
        home_elo = elo_ratings_dict[home_team] + 65
        away_elo = elo_ratings_dict[away_team]
        
        # check if home or away team won
        if score_diff < 0: 
            new_home_elo = elo_update(home_elo,away_elo,mov)[0] - 65
            new_away_elo = elo_update(home_elo,away_elo,mov)[1]
        elif score_diff > 0:
            new_away_elo = elo_update(away_elo,home_elo,mov)[0]
            new_home_elo = elo_update(away_elo,home_elo,mov)[1] - 65 
            
        # update the ratings dict
        elo_ratings_dict[home_team] = new_home_elo
        elo_ratings_dict[away_team] = new_away_elo
            
    elif row['location'] == 0.0:
        # since neutral just arbitrarily make home and away, but don't add 65
        home_team = row['team']
        away_team = row['opponent']
        
        # get corresponding elo ratings
        home_elo = elo_ratings_dict[home_team]
        away_elo = elo_ratings_dict[away_team]
        
        # check if home or away team won
        if score_diff > 0: 
            new_home_elo = elo_update(home_elo,away_elo,mov)[0]
            new_away_elo = elo_update(home_elo,away_elo,mov)[1]
        elif score_diff < 0:
            new_away_elo = elo_update(away_elo,home_elo,mov)[0]
            new_home_elo = elo_update(away_elo,home_elo,mov)[1] 
            
        # update the ratings dict
        elo_ratings_dict[home_team] = new_home_elo
        elo_ratings_dict[away_team] = new_away_elo
    
    
    

elo_ratings_dict

# See it in sorted form
import operator
sorted_elo_rankings = sorted(elo_ratings_dict.items(), key=operator.itemgetter(1))

sorted_elo_rankings

ivy_team_names = ['Yale','Harvard','Princeton','Cornell','Brown','Columbia','Dartmouth','Penn']
ivy_elo_ratings_dict = {}
for tpl in sorted_elo_rankings[::-1]:
    if tpl[0] in ivy_team_names:
        ivy_elo_ratings_dict[tpl[0]] = tpl[1]

ivy_elo_ratings_dict

