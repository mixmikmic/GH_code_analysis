import requests
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import seaborn as sns
import numpy as np
get_ipython().magic('matplotlib inline')
pd.set_option('display.max_columns',1000)

# set headers, otherwise the API may not return what we're looking for
HEADERS = {'user-agent': ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_5) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/45.0.2454.101 Safari/537.36'),
           'referer': 'http://stats.nba.com/scores/'
          }

# Get all Kobe shot data from 1996 to 2016 and put it into an array
# This player ID comes from stats.nba.com (http://stats.nba.com/player/#!/977/stats/)
playerID = "977"
seasons = []
for season in range(1996,2016):
    # The stats.nba.com API wants season as "1996-97"
    seasonString = str(season) + '-' + str(season+1)[2:]

    # The stats.nba.com endpoint we are using is http://stats.nba.com/stats/shotchartdetail
    # More info on endpoints: https://github.com/seemethere/nba_py/wiki/stats.nba.com-Endpoint-Documentation
    shot_chart_url = 'http://stats.nba.com/stats/shotchartdetail?Period=0&VsConference=&LeagueID=00&LastNGames=0&TeamID=0&Position=&Location=&Outcome=&ContextMeasure=FGA&DateFrom=&StartPeriod=&DateTo=&OpponentTeamID=0&ContextFilter=&RangeType=&Season=' + seasonString + '&AheadBehind=&PlayerID='+ playerID +'&EndRange=&VsDivision=&PointDiff=&RookieYear=&GameSegment=&Month=0&ClutchTime=&StartRange=&EndPeriod=&SeasonType=Regular+Season&SeasonSegment=&GameID='
    response = requests.get(shot_chart_url, headers=HEADERS)
    
    # Split response into headers and content
    headers = response.json()['resultSets'][0]['headers']
    shots = response.json()['resultSets'][0]['rowSet']

    # Create pandas dataframe to hold the data
    shot_df = pd.DataFrame(shots, columns=headers)

    # add extra column for season
    shot_df['SEASON'] = seasonString

    # add extra column for playoff flag
    shot_df['playoffs'] = 0
    
    seasons.append(shot_df)
    

# Do the same thing for all the playoff shots
for season in range(1996,2016):
    seasonString = str(season) + '-' + str(season+1)[2:]

    # This URL is the same except for the parameter SeasonType=Playoffs
    shot_chart_url = 'http://stats.nba.com/stats/shotchartdetail?CFID=&CFPARAMS=&ContextFilter=&ContextMeasure=FGA&DateFrom=&DateTo=&EndPeriod=10&EndRange=28800&GameID=&GameSegment=&LastNGames=0&LeagueID=00&Location=&Month=0&OpponentTeamID=0&Outcome=&Period=0&PlayerID='+ playerID +'&Position=&RangeType=0&RookieYear=&Season=' + str(season) + '-' + str(season+1)[2:] + '&SeasonSegment=&SeasonType=Playoffs&StartPeriod=1&StartRange=0&TeamID=1610612747&VsConference=&VsDivision='
    response = requests.get(shot_chart_url, headers=HEADERS)
    
    headers = response.json()['resultSets'][0]['headers']
    shots = response.json()['resultSets'][0]['rowSet']
    
    shot_df = pd.DataFrame(shots, columns=headers)
    shot_df['SEASON'] = str(season) + '-' + str(season+1)[2:]
    shot_df['playoffs'] = 1
    
    seasons.append(shot_df)

# combine all season and playoffs dataframes into one dataframe
kobe_all_shots = pd.concat(seasons)

# dump a csv file
# kobe_all_shots.to_csv("kobe_all_shots.csv)

# This number is two shots lower than it should be because two shots are missing from the data
len(kobe_all_shots)

kobe_all_shots.head()

# Combine NBA's very specific shot type descriptions into broader categories for filtering
#combine shot types
kobe_all_shots['COMBINED_SHOT_TYPE'] = kobe_all_shots['ACTION_TYPE']

kobe_all_shots.replace(to_replace={
        'COMBINED_SHOT_TYPE': {
            '(.+)?Jump (.+)?(S|s)hot':'Jump Shot',
            '(.+)?Fadeaway(.+)?':'Jump Shot',
            '(.+)?Dunk Shot':'Dunk',
            '(.+)?Layup (S|s)hot': "Layup",
            '(.+)?Hook.+':"Hook Shot",
            '(.+)?Tip.+':"Tip Shot",
            '(.+)?Bank.+':"Bank Shot",
            '(.+)?Finger Roll.+':"Layup"
        }
    }, regex=True, inplace=True)
kobe_all_shots.COMBINED_SHOT_TYPE.value_counts()

# Draw the court — this code comes from Savvas Tjortjoglou (http://savvastjortjoglou.com/nba-shot-sharts.html)
from matplotlib.patches import Circle, Rectangle, Arc

def draw_court(ax=None, color='black', lw=2, outer_lines=False):
    # If an axes object isn't provided to plot onto, just get current one
    if ax is None:
        ax = plt.gca()

    # Create the various parts of an NBA basketball court

    # Create the basketball hoop
    # Diameter of a hoop is 18" so it has a radius of 9", which is a value
    # 7.5 in our coordinate system
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

    # Create backboard
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)

    # The paint
    # Create the outer box 0f the paint, width=16ft, height=19ft
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                          fill=False)
    # Create the inner box of the paint, widt=12ft, height=19ft
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                          fill=False)

    # Create free throw top arc
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    # Create free throw bottom arc
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    # Restricted Zone, it is an arc with 4ft radius from center of the hoop
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color)

    # Three point line
    # Create the side 3pt lines, they are 14ft long before they begin to arc
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                               color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
    # I just played around with the theta values until they lined up with the 
    # threes
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                    color=color)

    # Center Court
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                           linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    return ax

# plot all the shots
sns.set_style("white")
sns.set_color_codes()
all_shot_fig = plt.figure(figsize=(25,47),dpi=72)
all_shot_ax = all_shot_fig.add_subplot(111)

all_shot_ax.set_ylim([-100,840])
all_shot_ax.set_xlim([250,-250])

draw_court(ax=all_shot_ax,outer_lines=True)

# First, draw the missed shots
all_shot_ax.scatter(kobe_all_shots[
        (kobe_all_shots.EVENT_TYPE == "Missed Shot")
    ].LOC_X,
    kobe_all_shots[
        (kobe_all_shots.EVENT_TYPE == "Missed Shot")
    ].LOC_Y,color='#d8b055',alpha=0.5)

# Then the made shots
all_shot_ax.scatter(kobe_all_shots[
        (kobe_all_shots.EVENT_TYPE == "Made Shot")
    ].LOC_X,
    kobe_all_shots[
        (kobe_all_shots.EVENT_TYPE == "Made Shot")
    ].LOC_Y,color='#6a3a89',alpha=0.5)

# save an svg of each season
#     all_shot_fig.savefig('shotchart_' + season_string + '.svg')

# Get all Kobe game logs from 1996 to 2016
seasons_games = []
for season in range(1996,2016):
    # get regular season game logs
    gamelog_url = 'http://stats.nba.com/stats/playergamelog?LeagueID=00&PerMode=PerGame&PlayerID='+ playerID +'&Season='+str(season) + '-' + str(season+1)[2:]+'&SeasonType=Regular+Season'
    response = requests.get(gamelog_url, headers=HEADERS)
    
    headers = response.json()['resultSets'][0]['headers']
    gamelogs = response.json()['resultSets'][0]['rowSet']
    gamelog_df = pd.DataFrame(gamelogs, columns=headers)
    gamelog_df['SEASON'] = str(season) + '-' + str(season+1)[2:]
    
    
    seasons_games.append(gamelog_df)
    
    # get playoff game logs
    gamelog_url = 'http://stats.nba.com/stats/playergamelog?LeagueID=00&PerMode=PerGame&PlayerID='+ playerID +'&Season='+str(season) + '-' + str(season+1)[2:]+'&SeasonType=Playoffs'
    response = requests.get(gamelog_url, headers=HEADERS)
    
    headers = response.json()['resultSets'][0]['headers']
    gamelogs = response.json()['resultSets'][0]['rowSet']
    gamelog_df = pd.DataFrame(gamelogs, columns=headers)
    gamelog_df['SEASON'] = str(season) + '-' + str(season+1)[2:]
    seasons_games.append(gamelog_df)

kobe_game_logs = pd.concat(seasons_games)

# Grab opponent from matchuip and condense some opponent names
kobe_game_logs["OPPONENT"] = kobe_game_logs["MATCHUP"].str[-3:]
kobe_game_logs.replace(to_replace={
        'OPPONENT': {
            "CHH":"CHA",
            "NOK":"NOP",
            "PHO":"PHX",
            "SAN":"SAS",
            "UTH":"UTA"
        }
    }, regex=True, inplace=True)
kobe_game_logs.OPPONENT.value_counts().sort_index()

# save to csv
# kobe_game_logs.to_csv("kobe_game_logs.csv")





