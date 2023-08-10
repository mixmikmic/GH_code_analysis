import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from time import sleep
pd.options.display.max_columns = 30

stad = pd.read_csv('../data/MLB-stadium-capacity.csv')

teams = stad.Team.unique()
teams

# angels: CAL 1990-1996, ANA 1997-2004, LAA 2005-2016
# marlins: FLA 1993-2011, MIA 2012-2016
# tampa bay: TBD 1998-2007, TBR 2008-2016
# montreal expos no longer a team

# drop at
# drop boxscore
# day or night into dummy variable
## double header dummy variable
## opening day dummy variable
# convert gb to float
# split record into wins and losses columns
## runs moving average
## runs allowed moving average
# convert streak to integers
# walk off dummy?

def get_games():
    team_df = pd.DataFrame(columns=['date', 'box', 'team', 'at', 'opponent', 'w_or_l', 'runs', 'runs_allowed',
                                    'innings', 'record', 'div_rank', 'gb', 'winning_pitcher', 'losing_pitcher',
                                    'save', 'time', 'd_or_n', 'attendance', 'streak', 'double_header', 'runs_pg',
                                    'runs_ma', 'runs_allowed_ma', 'ten_game', 'opening_day'])
    for team in teams:
        for year in range (1990, 2017):
            try:
                df = pd.DataFrame(columns=['date', 'box', 'team', 'at', 'opponent', 'w_or_l', 'runs', 'runs_allowed',
                                           'innings', 'record', 'div_rank', 'gb', 'winning_pitcher', 'losing_pitcher',
                                           'save', 'time', 'd_or_n', 'attendance', 'streak'])
                html = requests.get('http://www.baseball-reference.com/teams/' + team + '/' + str(year) +
                                    '-schedule-scores.shtml')
                bs = BeautifulSoup(html.text, 'html.parser')
                for game in bs.find('table', {'class':'stats_table'}).find_all('tr'):
                    results = []
                    for element in game.find_all('td'):
                        results.append(element.text)
                    if len(results) == 19:
                        df.loc[len(df)] = results

                # add double header dummy variable
                dh = '('
                df['double_header'] = [1 if dh in x else 0 for x in df.date]

                # adjust date and convert to datetime
                df.date = df.date.str.replace(r"\(.*\)","")
                date = []
                for i in df.date:
                    split = i.split(', ')
                    date.append(split[1] + ', ' + str(year))
                df.date = date
                df.date = pd.to_datetime(df.date)

                # shift record, div_rank, gb, streak and fill in the first value
                df.record = df.record.shift()
                df.record.fillna('0-0', inplace=True)

                for col in ['div_rank', 'gb', 'streak']:
                    df[col] = df[col].shift()
                    df[col].fillna('0', inplace=True)

                # rolling means for runs and runs allowed
                df['runs_pg'] = [0 if x == min(df.date) else df[df.date < x].runs.astype(int).mean() for x in df.date]
                df['runs_ma'] = df.runs.rolling(5).mean().shift()
                df['runs_allowed_ma'] = df.runs_allowed.rolling(5).mean().shift()
                df.runs_allowed_ma.fillna(df.iloc[5].runs_allowed_ma, inplace=True)
                
                # win_pct of last ten games
                ten_game = [list(df.w_or_l[i-10:i]).count('W') / 10. for i in range(10, len(df))]
                for i in range(10):
                    ten_game.insert(0, None)
                df['ten_game'] = ten_game

                # filter down to only home games
                df = df[~df['at'].str.contains('@')].reset_index(drop=True)

                # opening day dummy variable
                df['opening_day'] = [1 if x == min(df.date) else 0 for x in df.date]

                # create one major df
                team_df = pd.concat([team_df, df]).reset_index(drop=True)

                # sleep
                sleep(0.5)
                
            except:
                pass
    
    # save it to csv
    team_df.to_csv('../data/mlb-game-data-1990-2016.csv', index=False, encoding='utf-8')
    return team_df

get_ipython().magic('time all_teams = get_games()')

def get_season_info():
    complete_df = pd.DataFrame(columns = ['year', 'team', 'team_name', 'division', 'G', 'W', 'L', 'ties', 'win_pct',
                                          'p_win_pct', 'finish', 'gb', 'playoffs', 'R', 'RA', 'attendance', 'bat_age',
                                          'pitch_age', 'num_bat', 'num_pitch', 'top_player', 'managers'])
    for team in teams:
        try:
            df = pd.DataFrame(columns = ['year', 'team', 'team_name', 'division', 'G', 'W', 'L', 'ties', 'win_pct',
                                         'p_win_pct', 'finish', 'gb', 'playoffs', 'R', 'RA', 'attendance', 'bat_age',
                                         'pitch_age', 'num_bat', 'num_pitch', 'top_player', 'managers'])
            html = requests.get('http://www.baseball-reference.com/teams/' + team + '/')
            bs = BeautifulSoup(html.text, 'html.parser')
            year = 2017
            for season in bs.find('table', {'class':'sortable stats_table'}, {'id':'franchise_years'}).find_all('tr'):
                results = []
                for element in season.find_all('td'):        
                    results.append(element.text)
                if len(results) == 20:
                    results.insert(0, team)
                    results.insert(0, year)
                    df.loc[len(df)] = results
                    year -= 1
            
            # create one major df
            complete_df = pd.concat([complete_df, df])

            # sleep
            sleep(1)
        
        except:
            pass
    
    # filter df to 1990-2016
    complete_df = complete_df[(complete_df.year >= 1990) & (complete_df.year <= 2016)]
    
    # drop duplicates
    complete_df.drop_duplicates(inplace=True)

    # encode team and playoffs columns
    for col in ['team_name', 'playoffs', 'managers']:
        complete_df[col] = complete_df[col].str.encode('utf-8').str.replace('\xc2\xa0', ' ')
    
    # save it to csv
    complete_df.to_csv('../data/franchise-yearly-data.csv', index=False, encoding='utf-8')
    
    return complete_df

get_ipython().magic('time franchise = get_season_info()')

franchise.head()

franchise.shape

franchise.playoffs.unique()

