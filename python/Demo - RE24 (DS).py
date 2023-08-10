get_ipython().run_line_magic('run', '../../utils/notebook_setup.py')

from datascience import Table
from datascience.util import table_apply

import numpy as np

# custom functions that will help do some simple tasks
from datascience_utils import *
from datascience_stats import *
from datascience_topic import fast_run_expectancy, most_common_lineup_position

cols = ['EVENT_ID', 'INN_CT', 'EVENT_CD', 'OUTS_CT', 'BAT_ID', 'BAT_LINEUP_ID',
        'BAT_EVENT_FL', 'START_BASES_CD', 'END_BASES_CD', 'EVENT_OUTS_CT',
        'EVENT_RUNS_CT', 'FATE_RUNS_CT']
retro = Table.read_table('retrosheet_events-2001.csv.gz', sep=',', usecols=cols)

new_cols = ['ID', 'Inning', 'Event_Type', 'Outs', 'Batter_ID', 'Lineup_Order',
            'PA_Flag', 'Start_Bases', 'End_Bases', 'Event_Outs', 'Event_Runs',
            'Future_Runs']
retro.relabel(cols, new_cols)

bat_mask = (retro['PA_Flag'] == "T")
retro = retro.where(bat_mask).copy()

inning_mask = (retro['Inning'] < 9)
retro = retro.where(inning_mask).copy()

base_runner_codes = {
    0: "None on",  # No one on
    1: "1st",  # runner on 1st
    2: "2nd",  # runner on 2nd
    3: "1st and 2nd",  # runners on 1st & 2nd
    4: "3rd",  # runner on 3rd
    5: "1st and 3rd",  # runners on 1st & 3rd
    6: "2nd and 3rd",  # runners on 2nd & 3rd
    7: "Bases Loaded"  # bases loaded
}
# Replace the numeric code with a string code
retro['Start_Bases'] = replace(retro, 'Start_Bases', base_runner_codes)
retro['End_Bases'] = replace(retro, 'End_Bases', base_runner_codes)

event_codes = {
    0: 'Unknown',
    1: 'None',
    2: 'Generic out',
    3: 'K',  # Strikeout
    4: 'SB',  # Stolen Base
    5: 'Defensive indifference',
    6: 'CS',  # Caught stealing
    7: 'Pickoff error',
    8: 'Pickoff',
    9: 'Wild pitch',
    10: 'Passed ball',
    11: 'Balk',
    12: 'Other advance/out advancing',
    13: 'Foul error',
    14: 'BB',  # Walk
    15: 'IBB',  # Intentional walk
    16: 'HBP',  # Hit by pitch
    17: 'Interference',
    18: 'RBOE',  # Reached base on error
    19: 'FC',  # Fielder's choice
    20: '1B',  # Single
    21: '2B',  # Double
    22: '3B',  # Triple
    23: 'HR',  # Home run
    24: 'Missing play',
}

# Replace numeric code with string
retro['Event_Type'] = replace(retro, 'Event_Type', event_codes)

retro.show(10)

re = Table.read_table('re_2001.csv', sep=',')

# NOTE: THIS CELL IS SLOW.
# It's easier to understand what the calculation is with this block of code but
# after that, you might as well run the next cell.

# # Iterate through PAs computing run expectancy
# run_exp = np.array([
#     re.where('Outs', outs).\
#         where('Start_Bases', base)['RE'].\
#         item() if outs < 3 else 0
#     for outs, base in zip(retro['Outs'], retro['Start_Bases'])
# ])
# retro['Run_Expectancy'] = run_exp

# # Iterate through PAs computing run expectancy for next PA
# next_outs = retro['Outs'] + retro['Event_Outs']
# next_run_exp = np.array([
#     re.where('Outs', outs).\
#         where('Start_Bases', base)['RE'].\
#         item() if outs < 3 else 0
#     for outs, base in zip(next_outs, retro['End_Bases'])
# ])
# retro['Run_Expectancy_Next'] = next_run_exp

# This only takes about a second or two
retro = fast_run_expectancy(retro, re)

view_cols = ['Batter_ID', 'Lineup_Order', 'Inning', 'Outs', 'Event_Type',
             'Start_Bases', 'End_Bases', 'Event_Runs', 'Run_Expectancy',
             'Run_Expectancy_Next']
retro.select(*[view_cols]).show(5)

retro['RE24'] = retro['Run_Expectancy_Next'] -     retro['Run_Expectancy'] + retro['Event_Runs']

retro.select(*(view_cols + ['RE24'])).show(10)

retro.select(*(view_cols + ['RE24'])).take(23080)

retro.select(*(view_cols + ['RE24'])).take(9090)

retro.select(*(view_cols + ['RE24'])).take(4097)

retro.select(*(view_cols + ['RE24'])).take[[112064, 112065]]

retro.select(*(view_cols + ['RE24'])).take[[26273, 26274]]

batter_data = retro.select('Batter_ID', 'Run_Expectancy', 'RE24').    group('Batter_ID', collect=sum)
batter_data.relabel(
    ['Run_Expectancy sum', 'RE24 sum'], ['Run_Expectancy', 'RE24'])
batter_data.sort('RE24', descending=True).show(10)

# Helper function to do fast groupby and find the most common 
# lineup position for each batter
lineup_pos = most_common_lineup_position(retro)













if 'Lineup_Order' in batter_data.labels:
    re_per_pa = batter_data['Run_Expectancy'] / batter_data['PA']
    re24_per_pa = batter_data['RE24'] / batter_data['PA']
    batter_data = batter_data.with_columns(
        'RE per PA', re_per_pa,
        'RE24 per PA', re24_per_pa
    )
    qualified_batters = batter_data.where('qual', 'Qualified')
    # boxplot_pd is a helper function to use Pandas-style boxplot visualizations
    boxplots(qualified_batters, column='RE per PA', by='Lineup_Order')
    boxplots(qualified_batters, column='RE24 per PA', by='Lineup_Order')

if 'Lineup_Order' in batter_data.labels:
    qualified_batters.sort('RE24 per PA', descending=True).show()

