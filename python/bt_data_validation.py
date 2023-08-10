get_ipython().run_line_magic('matplotlib', 'inline')
from psycopg2 import connect
import psycopg2.sql as pg
import configparser
from datetime import datetime, timedelta, date
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
from matplotlib.ticker import FuncFormatter
import pandas as pd
import pandas.io.sql as pandasql
import seaborn as sns
from IPython.display import HTML
def print_table(sql, con):
    return HTML(pandasql.read_sql(sql, con).to_html(index=False))

# setting up pgsql connection
CONFIG = configparser.ConfigParser()
CONFIG.read('db.cfg')
dbset = CONFIG['DBSETTINGS']
con = connect(**dbset)

sql = pg.SQL('SELECT * FROM ryu4.aggr_bt_directional')
direction_obs = pandasql.read_sql(sql, con)

groups = direction_obs.groupby('street')
colours = np.random.rand(len(groups), 3,)
colour_cycle = cycle(colours)
fig, ax = plt.subplots(1, 1, figsize=(16,9))
ax.grid(color='silver', linestyle='-', linewidth=1)
ax.set_xlabel('EB/NB BT Observations')
ax.set_ylabel('WB/SB BT Observations')

for i, (street, data) in enumerate(groups):
    ax.plot(data.eb_nb_obs, data.wb_sb_obs, marker='o', markersize=10, alpha=0.7, label=street, linestyle='', color=next(colour_cycle))
plt.legend()
plt.show()

sql = '''SELECT d.sb_wb_report_name AS "WB/SB Route Name",         d.nb_eb_report_name AS "EB/NB Route Name",         d.eb_nb_obs AS "Number EB/NB obs",         d.wb_sb_obs AS "Number WB/SB obs",         CASE
            WHEN d.eb_nb_obs > d.wb_sb_obs THEN (d.eb_nb_obs * 1.0) / (d.eb_nb_obs + d.wb_sb_obs)
            WHEN d.eb_nb_obs < d.wb_sb_obs THEN (d.wb_sb_obs * 1.0) / (d.eb_nb_obs + d.wb_sb_obs)
        END AS "EB/WB NB/SB Ratio",
        CASE
            WHEN rp.direction = 'EBWB' THEN CASE
                                            WHEN d.eb_nb_obs < d.wb_sb_obs THEN 'WB'
                                            WHEN d.eb_nb_obs > d.wb_sb_obs THEN 'EB'
                                            END
            WHEN rp.direction = 'NBSB' THEN CASE
                                            WHEN d.eb_nb_obs < d.wb_sb_obs THEN 'SB'
                                            WHEN d.eb_nb_obs > d.wb_sb_obs THEN 'NB'
                                            END
        END AS "Bias Towards"
        FROM ryu4.aggr_bt_directional d
        INNER JOIN ryu4.bt_route_pairs rp ON rp.nb_eb_report_name = d.nb_eb_report_name 
        ORDER BY "EB/WB NB/SB Ratio" DESC;'''
print_table(sql, con)

# select one week of data between '2017-10-25 00:00:00' and '2017-11-01 00:00:00'
sql = pg.SQL('SELECT * FROM ryu4.aggr_bt_directional_oneweek')
direction_obs_oneweek = pandasql.read_sql(sql, con)

groups = direction_obs_oneweek.groupby('street')
fig, ax = plt.subplots(1, 1, figsize=(16,9))
ax.grid(color='silver', linestyle='-', linewidth=1)
ax.set_xlabel('EB/NB BT Observations')
ax.set_ylabel('WB/SB BT Observations')
for i, (street, data) in enumerate(groups):
    ax.plot(data.eb_nb_obs, data.wb_sb_obs, marker='o', markersize=10, alpha=0.7, label=street, linestyle='', color=next(colour_cycle))
plt.legend()
plt.show()

#EB/WB Observation ratios
sql = '''SELECT 
        CASE
            WHEN d.eb_nb_obs > d.wb_sb_obs THEN (d.eb_nb_obs * 1.0) / (d.eb_nb_obs + d.wb_sb_obs)
            WHEN d.eb_nb_obs < d.wb_sb_obs THEN (d.wb_sb_obs * 1.0) / (d.eb_nb_obs + d.wb_sb_obs)
        END AS "ratio"
        FROM ryu4.aggr_bt_directional_oneweek d
        INNER JOIN ryu4.bt_route_pairs rp ON rp.nb_eb_report_name = d.nb_eb_report_name 
        WHERE rp.direction = 'EBWB'
        ORDER BY "ratio" DESC;'''
ebwb_obs = pandasql.read_sql(sql, con)
#NB/SB Observation ratios
sql = '''SELECT 
        CASE
            WHEN d.eb_nb_obs > d.wb_sb_obs THEN (d.eb_nb_obs * 1.0) / (d.eb_nb_obs + d.wb_sb_obs)
            WHEN d.eb_nb_obs < d.wb_sb_obs THEN (d.wb_sb_obs * 1.0) / (d.eb_nb_obs + d.wb_sb_obs)
        END AS "ratio"
        FROM ryu4.aggr_bt_directional_oneweek d
        INNER JOIN ryu4.bt_route_pairs rp ON rp.nb_eb_report_name = d.nb_eb_report_name 
        WHERE rp.direction = 'NBSB'
        ORDER BY "ratio" DESC;'''
nbsb_obs = pandasql.read_sql(sql, con)
#EB Observation ratio
sql = '''SELECT 
        CASE
            WHEN d.eb_nb_obs > d.wb_sb_obs THEN (d.eb_nb_obs * 1.0) / (d.eb_nb_obs + d.wb_sb_obs)
            WHEN d.eb_nb_obs < d.wb_sb_obs THEN (d.wb_sb_obs * 1.0) / (d.eb_nb_obs + d.wb_sb_obs)
        END AS "ratio"
        FROM ryu4.aggr_bt_directional_oneweek d
        INNER JOIN ryu4.bt_route_pairs rp ON rp.nb_eb_report_name = d.nb_eb_report_name 
        WHERE rp.direction = 'EBWB' AND d.eb_nb_obs > d.wb_sb_obs
        ORDER BY "ratio" DESC;'''
eb_bias_obs = pandasql.read_sql(sql, con)
#WB Observation ratio
sql = '''SELECT 
        CASE
            WHEN d.eb_nb_obs > d.wb_sb_obs THEN (d.eb_nb_obs * 1.0) / (d.eb_nb_obs + d.wb_sb_obs)
            WHEN d.eb_nb_obs < d.wb_sb_obs THEN (d.wb_sb_obs * 1.0) / (d.eb_nb_obs + d.wb_sb_obs)
        END AS "ratio"
        FROM ryu4.aggr_bt_directional_oneweek d
        INNER JOIN ryu4.bt_route_pairs rp ON rp.nb_eb_report_name = d.nb_eb_report_name 
        WHERE rp.direction = 'EBWB' AND d.eb_nb_obs < d.wb_sb_obs
        ORDER BY "ratio" DESC;'''
wb_bias_obs = pandasql.read_sql(sql, con)
#NB Observation ratio
sql = '''SELECT 
        CASE
            WHEN d.eb_nb_obs > d.wb_sb_obs THEN (d.eb_nb_obs * 1.0) / (d.eb_nb_obs + d.wb_sb_obs)
            WHEN d.eb_nb_obs < d.wb_sb_obs THEN (d.wb_sb_obs * 1.0) / (d.eb_nb_obs + d.wb_sb_obs)
        END AS "ratio"
        FROM ryu4.aggr_bt_directional_oneweek d
        INNER JOIN ryu4.bt_route_pairs rp ON rp.nb_eb_report_name = d.nb_eb_report_name 
        WHERE rp.direction = 'NBSB' AND d.eb_nb_obs > d.wb_sb_obs
        ORDER BY "ratio" DESC;'''
nb_bias_obs = pandasql.read_sql(sql, con)
#SB Observation ratio
sql = '''SELECT 
        CASE
            WHEN d.eb_nb_obs > d.wb_sb_obs THEN (d.eb_nb_obs * 1.0) / (d.eb_nb_obs + d.wb_sb_obs)
            WHEN d.eb_nb_obs < d.wb_sb_obs THEN (d.wb_sb_obs * 1.0) / (d.eb_nb_obs + d.wb_sb_obs)
        END AS "ratio"
        FROM ryu4.aggr_bt_directional_oneweek d
        INNER JOIN ryu4.bt_route_pairs rp ON rp.nb_eb_report_name = d.nb_eb_report_name 
        WHERE rp.direction = 'NBSB' AND d.eb_nb_obs < d.wb_sb_obs
        ORDER BY "ratio" DESC;'''
sb_bias_obs = pandasql.read_sql(sql, con)

# Box plot comparing EB/WB vs NB/SB ratios
data = [ebwb_obs, nbsb_obs]
plt.subplot(1,1,1)
plt.boxplot(data)
plt.xticks([1,2],['East/West Bound', 'North/South Bound'])
plt.title("Box Plot of Ratios for East/West Bound and North/South Bound BT Readings")
plt.show()

# Box plots comparing EB vs WB and NB vs SB ratios
data = [eb_bias_obs, wb_bias_obs]
ax1 = plt.subplot(121)
plt.boxplot(data)
plt.xticks([1,2],['East Bound', 'West Bound'])
plt.title("East/West Bound Bias")
data = [nb_bias_obs, sb_bias_obs]
ax2 = plt.subplot(122, sharey=ax1)
plt.boxplot(data)
plt.xticks([1,2],['North Bound', 'South Bound'])
plt.title("North/South Bound Bias")
plt.tight_layout()
plt.show()

sql = '''SELECT sb_wb_report_name AS "wbsb_route",         nb_eb_report_name AS "ebnb_route"         FROM ryu4.aggr_bt_directional_oneweek
        ORDER BY eb_nb_obs - wb_sb_obs;'''
routes = pandasql.read_sql(sql, con)
ebnb_route = routes.ebnb_route
wbsb_route = routes.wbsb_route

# setting up 16 x 4 facet grid for day of week plots
# Note: There are only 62 routes, so there are 2 empty plots
fig, ax = plt.subplots(nrows=16, ncols=4, sharex= True, sharey= True, figsize=(16,48))
row, col = 0, 0 
for i in range(len(routes)):
    sql = pg.SQL('''SELECT eb_nb_obs, wb_sb_obs, date_bin, street, from_street, to_street             FROM ryu4.aggr_bt_directional_day             WHERE nb_eb_report_name = {ebnb_route} AND sb_wb_report_name = {wbsb_route}
            ORDER BY date_bin''').format(ebnb_route = pg.Literal(ebnb_route[i]), wbsb_route = pg.Literal(wbsb_route[i]))
    route_obs_day = pandasql.read_sql(sql, con)
    # setting up dual bar graphs
    N = len(route_obs_day)
    ind = np.arange(N)
    width = 0.35
    ebnb_rects = ax[row,col].bar(ind, route_obs_day.eb_nb_obs, width, color='orange')
    wbsb_rects = ax[row,col].bar(ind + width, route_obs_day.wb_sb_obs, width, color='green')
    ax[row,col].set_ylabel('Observations')
    ax[row,col].set_title(route_obs_day.street[0] + ' - ' + 
                          route_obs_day.from_street[0] + ' to ' +
                          route_obs_day.to_street[0])
    ax[row,col].set_xticks(ind + width / 2)
    ax[row,col].set_xticklabels([date.strftime("%A %x") for date in route_obs_day.date_bin], rotation=90)
    ax[row,col].legend((ebnb_rects[0], wbsb_rects[0]), ('EB/NB', 'WB/SB'))
    col += 1
    if col%4 == 0: 
        row += 1
        col  = 0
print('Day of Week Plots for BT Observations')
plt.show()

sql = '''SELECT sb_wb_report_name AS "wbsb_route",         nb_eb_report_name AS "ebnb_route"         FROM ryu4.aggr_bt_directional_oneweek
        ORDER BY CASE
            WHEN eb_nb_obs > wb_sb_obs THEN (eb_nb_obs * 1.0) / (eb_nb_obs + wb_sb_obs)
            WHEN eb_nb_obs < wb_sb_obs THEN (wb_sb_obs * 1.0) / (eb_nb_obs + wb_sb_obs)
        END DESC;'''
routes = pandasql.read_sql(sql, con)
ebnb_route = routes.ebnb_route
wbsb_route = routes.wbsb_route

for i in range(len(routes)):
    print(ebnb_route[i] + " / " + wbsb_route[i])
    sql = pg.SQL('''CREATE OR REPLACE TEMP VIEW wbsb AS                     SELECT DATE_TRUNC('hour', datetime_bin) datetime_bin, SUM(obs) obs                     FROM aggr_5min_bt                     WHERE report_name = {wbsb_route}                     AND datetime_bin < '2017-11-07' AND datetime_bin >= '2017-11-01'                     GROUP BY report_name, DATE_TRUNC('hour', datetime_bin);                     CREATE OR REPLACE TEMP VIEW ebnb AS                     SELECT DATE_TRUNC('hour', datetime_bin) datetime_bin, SUM(obs) obs                     FROM aggr_5min_bt                     WHERE report_name = {ebnb_route}                     AND datetime_bin < '2017-11-07' AND datetime_bin >= '2017-11-01'                     GROUP BY report_name, DATE_TRUNC('hour', datetime_bin);                     SELECT wbsb.datetime_bin datetime_bin, wbsb.obs wbsb_obs, ebnb.obs ebnb_obs
                    FROM wbsb \
                    INNER JOIN ebnb ON wbsb.datetime_bin = ebnb.datetime_bin \
                    ORDER BY datetime_bin;
        ''').format(ebnb_route = pg.Literal(ebnb_route[i]), wbsb_route = pg.Literal(wbsb_route[i]))
    route_obs = pandasql.read_sql(sql, con)
    fig, ax = plt.subplots(1, 1, figsize=(16,5))
    ax.grid(color='silver', linestyle='-', linewidth=1)
    ebnb_line = ax.plot(route_obs.datetime_bin, route_obs.ebnb_obs, color='orange')
    wbsb_line = ax.plot(route_obs.datetime_bin, route_obs.wbsb_obs, color='green')
    ax.set_ylabel('Observations')
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday = [0, 1, 2, 3, 4, 5, 6]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('\n%a %Y-%m-%d'))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval = 8))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%Hh'))
    ax.legend((ebnb_line[0], wbsb_line[0]), ('EB/NB', 'WB/SB'))
    plt.show()

# One way streets are ommited.
sql = pg.SQL('''SELECT DISTINCT segment FROM ryu4.bt_segments WHERE direction IN ('EB', 'WB')
AND segment NOT SIMILAR TO '(Richmond|Adelaide|Wellington|Eastern|Front/|Front Yonge)%' ORDER BY segment''')
ebwb_segments = pandasql.read_sql(sql, con)

fig, ax = plt.subplots(nrows=7, ncols=5, sharey= True, figsize=(16,48))
row, col = 0, 0
print("Speed Distribution of East/West Routes")
for i in range(len(ebwb_segments)):
    segment = ebwb_segments.segment[i]
    sql = pg.SQL('''SELECT seg.segment AS "Segment", trunc(3.6*seg.length_m/bt.travel_time) AS "Speed (kph)",
            seg.direction AS "Direction"
            FROM ryu4.aggr_15min_bt bt
            JOIN ryu4.bt_segments seg ON seg.analysis_id = bt.analysis_id
            WHERE bt.travel_time > 0 AND seg.direction IN ('EB', 'WB') AND seg.segment = {segment}
            GROUP BY segment, "Speed (kph)", seg.direction;''').format(segment = pg.Literal(segment))
    bt_speed_distrib = pandasql.read_sql(sql, con)
    sns.violinplot(ax=ax[row, col], x="Segment", y="Speed (kph)", hue="Direction", inner="quart",
               data = bt_speed_distrib[(bt_speed_distrib["Speed (kph)"] < 70) & (bt_speed_distrib["Speed (kph)"] > 0)],
               split=True, ylim=[0,70], cut=0)
    sns.set_style("darkgrid")
    col += 1
    if col%5 == 0: 
        row += 1
        col  = 0

sql = pg.SQL('''SELECT DISTINCT segment FROM ryu4.bt_segments WHERE direction IN ('NB', 'SB') ORDER BY segment''')
nbsb_segments = pandasql.read_sql(sql, con)

fig, ax = plt.subplots(nrows=6, ncols=5, sharey= True, figsize=(16,48))
row, col = 0, 0
print("Speed Distribution of North/South Routes")
for i in range(len(nbsb_segments)):
    segment = nbsb_segments.segment[i]
    sql = pg.SQL('''SELECT seg.segment AS "Segment", trunc(3.6*seg.length_m/bt.travel_time) AS "Speed (kph)",
            seg.direction AS "Direction"
            FROM ryu4.aggr_15min_bt bt
            JOIN ryu4.bt_segments seg ON seg.analysis_id = bt.analysis_id
            WHERE bt.travel_time > 0 AND seg.direction IN ('NB', 'SB') AND seg.segment = {segment}
            GROUP BY segment, "Speed (kph)", seg.direction;''').format(segment = pg.Literal(segment))
    bt_speed_distrib = pandasql.read_sql(sql, con)
    sns.violinplot(ax=ax[row, col], x="Segment", y="Speed (kph)", hue="Direction", inner="quart",
               data = bt_speed_distrib[(bt_speed_distrib["Speed (kph)"]< 70) & (bt_speed_distrib["Speed (kph)"] > 0)],
               split=True, ylim=[0,70], cut=0)
    sns.set_style("darkgrid")
    col += 1
    if col%5 == 0: 
        row += 1
        col  = 0



