import pandas as pd
import math

data = pd.read_csv('Data/2017_season_efficiencies.csv')
data

#NOT Scaled
def logit_team(stat_list):
    '''
    Order of stat list items: [OPASS, ORUSH, OINT, DPASS, DRUSH, TPEN]
    '''
    
    logit = (
        (0.55*stat_list[0]) +
        (0.52*stat_list[1]) -
        (9.53*stat_list[2]) -
        (0.54*stat_list[3]) -
        (0.58*stat_list[4]) -
        (0.59*stat_list[5])
    )
    return logit

#Scaled
def logit_team_scale(stat_list):
    '''
    Order of stat list items: [OPASS, ORUSH, OINT, DPASS, DRUSH, TPEN]
    '''
    
    logit = (
        (1.41*stat_list[0]) +
        (0.92*stat_list[1]) -
        (0.64*stat_list[2]) -
        (1.44*stat_list[3]) -
        (1.06*stat_list[4]) -
        (0.24*stat_list[5])
    )
    return logit

test = data.iloc[1].tolist()
del test[0]
test

logit_team(1, test)

ucf = data.iloc[99].tolist()
del ucf[0]

auburn = data.iloc[9].tolist()
del auburn[0]

ucf

logit_ucf = logit_team(ucf)
logit_aub = logit_team(auburn)

print(logit_ucf)
print(logit_aub)
print('------SCALED-------')

logit_ucf_s = logit_team_scale(ucf)
logit_aub_s = logit_team_scale(auburn)

print(logit_ucf_s)
print(logit_aub_s)

logit_game = 1.24 + 0.25*0 + logit_ucf - logit_aub
print(f'Not scaled: {logit_game}')
      
logit_game_s = -0.03 + 0.13*0 + logit_ucf_s - logit_aub_s
print(f'Scaled: {logit_game_s}')

odds = math.exp(logit_game)
odds_s = math.exp(logit_game_s)

print(f'Not scaled: {odds}')
print(f'Scaled: {odds_s}')

prob = odds/(1+odds)
prob_s = odds_s/(1+odds_s)

print(f'Not scaled UCF win prob: {prob}')
print(f'Scaled UCF win prob: {prob_s}')

print(1-prob)
print(1-prob_s)

data = data.set_index('TEAM')

data.head()

team_logit = []

for index, row in data.iterrows():
    stats = row.tolist()
    team_logit.append(logit_team_scale(stats))

data['LOGIT'] = team_logit
data = data.reset_index()
data.head()

output = data[['TEAM','LOGIT']]
output.head()

output.to_csv('Data/team_logits.csv', index=False)



