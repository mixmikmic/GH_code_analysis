import requests
import pandas as pd

BASE_URL = 'http://stats.nba.com/stats/{endpoint}'
HEADERS  = {'user-agent': ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_5) '
                           'AppleWebKit/537.36 (KHTML, like Gecko) '
                           'Chrome/45.0.2454.101 Safari/537.36')
           }

## helper function for requests
def get_json(endpoint, params, referer='scores'):
    """
    Args:
        endpoint (str): endpoint to be called from the API
        params (dict): parameters to be passed to the API
    Raises:
        HTTPError: if requests hits a status code != 200
    Returns:
        json (json): json object for selected API call
    """
    h = dict(HEADERS)
    h['referer'] = 'http://stats.nba.com/{ref}/'.format(ref=referer)
    response = requests.get(BASE_URL.format(endpoint=endpoint), params=params, headers=h)
    response.raise_for_status()
    return response.json()

ENDPOINT = 'leaguedashplayerstats'
PARAMS   = {'LeagueID': '00',       # NBA
            'Season': '2010-11',
            'SeasonType': 'Regular Season',
            'MeasureType': 'Base',  # options: Base, Advanced, Misc, Four Factors, Scoring, Opponent, Usage
            'PerMode': 'PerMinute', # options: PerGame, MinutesPer, PerMinute, PerPossession, ...
            'PlusMinus': 'N',       # ?
            'PaceAdjust': 'N',      # ?
            'Rank': 'N',
            'PORound': '0',         # all playoff rounds, other values pick specific rounds
            'Outcome': '',          # possible to filter by win ('W') or loss ('L') 
            'Location': '',         # possible to filter by 'Home' or 'Away'
            'Month': '0',           # all months, possible to filter by Oct ('1'), Nov ('2'), Dec ('3'), etc...
            'SeasonSegment': '',    # blank uses entire season
            'DateFrom': '',         # begin of date range filter
            'DateTo': '',           # end of date range filter
            'OpponentTeamID': '0',  # all opponents, or filter stats against specific teams
            'VsConference': '',
            'VsDivision': '',
            'TeamID': '0',
            'Conference': '',
            'Division': '',
            'GameSegment': '',      # all segments, other options: 'First Half', 'Second Half', 'Overtime'
            'Period': '0',          # can specifie quarter '1' or overtime period str(4+n)
            'ShotClockRange': '',
            'LastNGames': '0',
            'GameScope': '',
            'PlayerExperience': '',
            'PlayerPosition': '',
            'StarterBench': '',     # can select between 'Starter' or 'Bench' players
            'DraftYear': '',
            'DraftPick': '',
            'College': '',
            'Country': '',
            'Height': '',
            'Weight': ''
           }

# make the request and pull out the headers and rows
result = get_json(endpoint=ENDPOINT, params=PARAMS, referer='')
cols   = result['resultSets'][0]['headers']
rows   = result['resultSets'][0]['rowSet']

# restrict feature set to 7 key stats along with player/team info
features = [
 'PLAYER_ID',
 'PLAYER_NAME',
 'TEAM_ID',
 'TEAM_ABBREVIATION',
 'REB',
 'AST',
 'TOV',
 'STL',
 'BLK',
 'PF',
 'PTS'
]

# stats dataframe
df_stats = pd.DataFrame(rows, columns=cols)
df_stats = df_stats[df_stats.MIN > 0][features]
df_stats.head(5)

# helper function to pull down player details from the API
def get_player_details(player_id, season, fields=['POSITION']):
    # make request
    result  = get_json('commonplayerinfo', params={'PlayerID':player_id})['resultSets'][0]
    allrows = pd.DataFrame(result['rowSet'], columns=result['headers'])
    
    # return the first row on the requested fields in a dictionary
    return dict(allrows[(allrows['FROM_YEAR'] <= season) & (season <= allrows['TO_YEAR'])].ix[0,fields])

# build a dataframe of player details
df_full = df_stats.merge(df_stats.PLAYER_ID.apply(lambda s: pd.Series(get_player_details(s, 2010, ['POSITION', 'WEIGHT', 'HEIGHT']))), 
                         left_index=True, 
                         right_index=True)
df_full.head(5)

df_encode = df_full[['POSITION','TEAM_ABBREVIATION']]

# note -- positions are sometimes blank and sometimes contains two, e.g. forward-guard.
# for blank entries, label unknown. for multiple positions, assume the first one is more achetypical.
df_encode['POSITION'] = df_encode.POSITION.apply(lambda x: 'UNKNOWN' if x == '' else x.split('-')[0].upper())

# encode labels
df_encode = pd.get_dummies(df_encode, columns=['POSITION','TEAM_ABBREVIATION'], prefix={'POSITION':'POSITION', 'TEAM_ABBREVIATION':'TEAM'})

# merge back stats
df_encode = df_full.merge(df_encode, left_index=True, right_index=True)
df_encode.head(5)

import mapper

from scipy.spatial.distance import pdist
from sklearn import decomposition

# point cloud (just the stats)
pcd = df_stats[['PTS','REB','AST','STL','BLK','PF','TOV']].as_matrix()

# the metric parameters below specify variance-normalized Euclidean (seuclidean) for the
# dissimilarity metric, where the variance (V) is computed automatically
dist = pdist(pcd, metric='seuclidean')

# compute filter values using first & second SVD components
filt = mapper.filters.dm_eigenvector(data=pcd, k=[0,1], metricpar={})

# assign the cover for the filter functions for two resolutions
part_low_res  = mapper.cover.cube_cover_primitive(intervals=20, overlap=50)(filt)
part_high_res = mapper.cover.cube_cover_primitive(intervals=30, overlap=50)(filt)

# compute the mapper output (note: using single-linkage clustering default)
print('\n')
result_low_res  = mapper.mapper(dist, filt, part_low_res,  cutoff=mapper.cutoff.biggest_gap(), metricpar={}, verbose=False)
print('\n')
result_high_res = mapper.mapper(dist, filt, part_high_res, cutoff=mapper.cutoff.biggest_gap(), metricpar={}, verbose=False)

# import visualization utilities from parent directory
import sys

pwd = get_ipython().getoutput('pwd')
parent_dir = '/'.join(pwd[0].split('/')[:-1])
if parent_dir not in sys.path:
    sys.path.append('/'.join(pwd[0].split('/')[:-1]))

import d3_lib
import tda_mapper_extensions

from IPython.core.display import HTML

# low resolution graphs
G_pts_low_res  = tda_mapper_extensions.custom_d3js_fdgraph(result_low_res,  df_encode, feature='PTS')
G_reb_low_res  = tda_mapper_extensions.custom_d3js_fdgraph(result_low_res,  df_encode, feature='REB')

# high resolution graphs
G_pts_high_res = tda_mapper_extensions.custom_d3js_fdgraph(result_high_res, df_encode, feature='PTS')
G_reb_high_res = tda_mapper_extensions.custom_d3js_fdgraph(result_high_res, df_encode, feature='REB')

HTML(d3_lib.set_styles('force_directed') +
     '<script src="http://d3js.org/d3.v3.min.js"></script>' +
     '<script src="http://marvl.infotech.monash.edu/webcola/cola.v3.min.js"></script>' +
     d3_lib.draw_graph('force_directed_nba', {'data': G_pts_low_res}))

HTML(d3_lib.set_styles('force_directed') +
     '<script src="http://d3js.org/d3.v3.min.js"></script>' +
     '<script src="http://marvl.infotech.monash.edu/webcola/cola.v3.min.js"></script>' +
     d3_lib.draw_graph('force_directed_nba', {'data': G_reb_low_res}))

HTML(d3_lib.set_styles('force_directed') +
     '<script src="http://d3js.org/d3.v3.min.js"></script>' +
     '<script src="http://marvl.infotech.monash.edu/webcola/cola.v3.min.js"></script>' +
     d3_lib.draw_graph('force_directed_nba', {'data': G_pts_high_res}))

HTML(d3_lib.set_styles('force_directed') +
     '<script src="http://d3js.org/d3.v3.min.js"></script>' +
     '<script src="http://marvl.infotech.monash.edu/webcola/cola.v3.min.js"></script>' +
     d3_lib.draw_graph('force_directed_nba', {'data': G_reb_high_res}))

