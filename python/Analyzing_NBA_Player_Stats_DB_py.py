from db import DB
import pandas as pd

db = DB(filename="/home/pybokeh/Dropbox/data_sets/nba", dbtype="sqlite")

pd.set_option("display.max_columns",50)
pd.set_option("display.max_rows",999)

db.tables

sql = """
select
team.name_pos as name,
avgs.season as season,
avgs.fg_perc as fg_perc

from regular_season_avgs avgs

inner join 

(select
id,
name_pos

from player_game_stats

where
team_name like '%Cav%') as team
on team.id = avgs.id

where
not(season like '%14-''15%')

order by
team.name_pos,
avgs.season
"""

df = db.query(sql)
df

grouped = df['name'].value_counts()
grouped = grouped.sort_index()
new_index_list = [[value for value in range(1,index+1)] for index in grouped.values]

new_index = []
for mylist in new_index_list:
    new_index = new_index + mylist

df.index = new_index
df

df.reset_index(level=0, inplace=True)
df.rename(columns={'index':'season_num'}, inplace=True)
df

from ggplot import *

ggplot(df, aes(x='season_num', y='fg_perc', color='name')) +     ylab("FG %") +     xlab("Season Number") +     geom_line()

criteria = df['name'].str.contains('LeBron')
lebron = df[criteria]
lebron

criteria = df['name'].str.contains('Varejao')
av = df[criteria]
av.index = range(len(av))
av

criteria = df['name'].str.contains('Haywood')
bh = df[criteria]
bh

ggplot(lebron, aes(x='season_num', y='fg_perc', color='name')) +     geom_line()

lebron = lebron.reset_index(drop=True)
lebron

ggplot(lebron, aes(x='season_num', y='fg_perc', color='name')) +     geom_line()

criteria = df['name'].str.contains('Haywood')
bh = df[criteria]
bh = bh.reset_index(drop=True)
bh

ggplot(bh, aes(x='season_num', y='fg_perc', color='name')) +     geom_line()

criteria = df['name'].str.contains('Varejao')
av = df[criteria]
av = av.reset_index(drop=True)
av

ggplot(av, aes(x='season_num', y='fg_perc', color='name')) +     geom_line()

