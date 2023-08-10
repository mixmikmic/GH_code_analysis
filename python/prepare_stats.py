import pandas as pd

df = pd.read_csv('Data/test/2017_games.csv', parse_dates=[0])

df.head()

# Drop all games with FCS opponent

df = df.drop(df[df.Opp_Abbr == 'FCS'].index).reset_index(drop=True)
df.head()

# Match oppoenent's sacks and sack yardage to each team & game (sacks against)

sack_list = []
yard_list = []

for index, row in df.iterrows():
    try:
        date = row.Date
        opp = row.Opp_Abbr
        
        sacks = df.loc[(df['Date'] == date) & (df['Team_Abbr'] == opp), 'Def Sacks'].values[0]
        sack_list.append(sacks)
        
        yards = df.loc[(df['Date'] == date) & (df['Team_Abbr'] == opp), 'Def Sack Yards'].values[0]
        yard_list.append(yards)
        
        print(index)
        
    except IndexError:
        sacks.append('none')
        sack_yards.append('none')
        print(f'{index} : IndexError')

print(f'Dataframe length: {len(df)}')
print(f'sack_list length: {len(sack_list)}')
print(f'yard_list length: {len(yard_list)}')

# Add sacks and sack yards to dataframe

df['Off Sacks'] = sack_list
df['Off Sack Yds'] = yard_list

df.tail()

# Split Result column into win/loss outcome and score

df[['Result','Score']] = df['Result'].str.split(' ', expand=True)
df.head()

# Convert home field to binary

locations = df['Location'].tolist()
advantage = []

for site in locations:
    if site == 'home':
        advantage.append(1)
    elif site == 'away':
        advantage.append(0)
    elif site == 'neutral':
        advantage.append(0)

df['Home Adv'] = advantage

df.head(10)

df.to_csv('Data/testing_data.csv', index=False)

df1 = df

# Create efficiency stat functions

def off_pass(pass_yds, pass_att, sack_yds, sack_tot):
    try:
        n = pass_yds - sack_yds
        d = pass_att + sack_tot
        result = n/d
        return result
    except ZeroDivisionError:
        return 0

    
def off_rush(rush_yds, rush_att):
    try:
        result = rush_yds/rush_att
        return(result)
    except ZeroDivisionError:
        return 0
    

def off_int(int_tot, pass_att):
    try:
        result = int_tot/pass_att
        return result
    except ZeroDivisionError:
        return 0
    

def def_pass(pass_yds, pass_att):
    try:
        result = pass_yds/pass_att
        return result
    except ZeroDivisionError:
        return 0
    

def def_rush(rush_yds, rush_att):
    try:
        result = rush_yds/rush_att
        return result
    except ZeroDivisionError:
        return 0
    

def team_pen(pen_yds, play_tot):
    try:
        result = pen_yds/play_tot
        return result
    except ZeroDivisionError:
        return 0

# Calculate efficiency stats
OPASS_list = []
ORUSH_list = []
OINT_list = []
DPASS_list = []
DRUSH_list = []
TPEN_list = []

for index, row in df1.iterrows():
    OPASS_list.append(off_pass(
        row['Off Pass Yards'],
        row['Off Pass Att'],
        row['Off Sack Yds'],
        row['Off Sacks']
    ))
    ORUSH_list.append(off_rush(row['Off Rush Yards'], row['Off Rush Att']))
    OINT_list.append(off_int(row['Off Int'], row['Off Pass Att']))
    DPASS_list.append(def_pass(row['Def Pass Yards'], row['Def Pass Att']))
    DRUSH_list.append(def_rush(row['Def Rush Yards'], row['Def Rush Att']))
    TPEN_list.append(team_pen(row['Pen Yards'], row['Off Plays']))
    print(index)

print(len(OPASS_list))
print(len(ORUSH_list))
print(len(OINT_list))
print(len(DPASS_list))
print(len(DRUSH_list))
print(len(TPEN_list))

df1['OPASS'] = OPASS_list
df1['ORUSH'] = ORUSH_list
df1['OINT'] = OINT_list
df1['DPASS'] = DPASS_list
df1['DRUSH'] = DRUSH_list
df1['TPEN'] = TPEN_list

df1.head()

df1.to_csv('Data/testing_data_with_efficiency.csv', index=False)

eff_df = df1[['Result','Home Adv','OPASS','ORUSH','OINT','DPASS','DRUSH','TPEN']]
eff_df.head()

eff_df = eff_df.rename(columns={'Result':'LABEL', 'Home Adv':'HOME'})
eff_df.head()

eff_df.to_csv('Data/efficiency_stats_test.csv', index=False)

prod_df = pd.read_csv('Data/testing_data_with_efficiency.csv', parse_dates=[0])
prod_df.head()

prod_df = prod_df[['Team_Abbr','OPASS','ORUSH','OINT','DPASS','DRUSH','TPEN']]

prod_df = prod_df.groupby('Team_Abbr').mean().reset_index()
prod_df = prod_df.rename(columns={'Team_Abbr':'TEAM'})
prod_df

prod_df.to_csv('Data/2017_season_efficiencies.csv', index=False)



