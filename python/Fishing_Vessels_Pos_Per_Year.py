import csv
import bq
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sb
import matplotlib
import os

client = bq.Client.Get()

def Query(q):
    t0 = time.time()
    answer = client.ReadTableRows(client.Query(q)['configuration']['query']['destinationTable'])
    print 'Query time: ' + str(time.time() - t0) + ' seconds.'
    return answer

years = [2012,2013,2014,2015,2016]


pos_array = []

for y in years:
    y = str(y)
    startdate = y+'-01-01'
    enddate = y+'-12-31'

    q = '''
    select integer(positions/100)*100 positions_bin,
    count(*)
    from
    (SELECT
      a.mmsi AS mmsi,
      a.fishing_msg_ratio AS fishing_msg_ratio,
      b.c_pos as positions
    FROM (
      SELECT
        mmsi,
        COUNT(*) c_msg,
        sum (shiptype_text = 'Fishing') c_fishing,
        sum (shiptype_text = 'Fishing') / COUNT(*) fishing_msg_ratio
      FROM (TABLE_DATE_RANGE([pipeline_740__classify.], TIMESTAMP('{startdate}'), TIMESTAMP('{enddate}')))
      WHERE
        type IN (5,
          24)
        AND shiptype_text IS NOT NULL
        AND shiptype_text != 'Not available'
      GROUP EACH BY
        mmsi
      HAVING
        c_fishing > 10
        AND fishing_msg_ratio > .99 ) a
    JOIN EACH (
      SELECT
        INTEGER(mmsi) AS mmsi,
        COUNT(*) AS c_pos
      FROM (TABLE_DATE_RANGE([pipeline_740__classify.], TIMESTAMP('{startdate}'), TIMESTAMP('{enddate}')))
      WHERE
        lat IS NOT NULL
        AND lon IS NOT NULL
      GROUP BY
        mmsi
      HAVING
        c_pos > 100 )b
    ON
      a.mmsi = b.mmsi) group by positions_bin
      order by positions_bin asc
    '''.format(startdate=startdate, enddate=enddate)

    pos = Query(q)
    
    vs = np.array([int(p[1]) for p in pos])
    ps = np.array([int(p[0]) for p in pos])
    pos_array.append([ps,vs])

for pos, y in zip(pos_array, years):
    plt.loglog(pos[0],pos[1], label=str(y))
plt.ylabel("number of vessels per 100 positions")
plt.xlabel("Positions")
plt.title("number of positions vs. vessels")
plt.legend()
plt.show()

lessthan = []
greaterhthan = []
for pos in pos_array:
    a = 0 # less than 1000
    b = 0 # more than 1000
    for p,v in zip(pos[0],pos[1]):
        if p<1000:
            a+=v
        else:
            b+=v
    greaterhthan.append(b)
    lessthan.append(a)
data = [[l,g] for l, g in zip(lessthan,greaterhthan)]
data
df = pd.DataFrame(data,index=[2012,2013,2014,2015,2016],columns=['<1000','>1000'])

df.head()

matplotlib.style.use('bmh')
plt.rcParams["figure.figsize"] = [8,4]

df.plot.bar(color=['#66A6D2','#04317C'])   
plt.title("Number of Vessels with > and < 1000 \nPositions per Year")

counts = []
for year in range(2012,2017):
    q = '''
    select count(*) from [scratch_david_mmsi_lists.500_cutoff_{}]'''.format(year)
    p = int(Query(q)[0][0])
    counts.append(p)

df2 = df.copy()
df2["Old Lists"] = df[">1000"]
df2["New Lists"] = counts

df2[['Old Lists','New Lists']].plot.bar(color=['#66A6D2','#04317C'])   
plt.title("New Versus Old Vessel Lists")

# see how fishing hours vary with number of points

tot_points_ = []
num_mmsi_ = []
fishing_hours_adjusted_ = []
fishing_hours_ = []
for y in years:
    q = '''

    SELECT
      sum(a.fishing_hours) fishing_hours,
      sum(a.fishing_hours_adjusted) fishing_hours_adjusted,
      exact_count_distinct(a.mmsi) num_mmsi,
      b.tot_points tot_points,
    FROM (
      SELECT
        mmsi,
        SUM(fishing_hours) fishing_hours,
        SUM(fishing_hours_adjusted) fishing_hours_adjusted
      FROM
        [scratch_david.{y}_mmsi_summaries_v2]
      WHERE
        mmsi IN (
        SELECT
          mmsi
        FROM
          scratch_david_mmsi_lists.likely_fishing_{y})
      GROUP BY
        mmsi ) a
    LEFT JOIN (
      SELECT
        mmsi,
        SUM(points) tot_points,
        SUM(active_points) tot_active_points,
        SUM(IF(active_points>=5,1,0)) active_days5,
        SUM(IF(active_points>=10,1,0)) active_days10,
        SUM(IF(active_points>=15,1,0)) active_days15,
        SUM(IF(active_points>=20,1,0)) active_days20,
      FROM
        [scratch_david.{y}_mmsi_summaries_v2]
        group by mmsi) b
    ON
      a.mmsi = b.mmsi
    group by tot_points
    order by tot_points asc
    '''.format(y=y)
    fp = Query(q)

    fishing_hours = np.array([float(f[0]) for f in fp])
    fishing_hours = np.array([fishing_hours[i:].sum() for i in range(len(fp))])
    fishing_hours_adjusted = np.array([float(f[1]) for f in fp])
    fishing_hours_adjusted = np.array([fishing_hours_adjusted[i:].sum() for i in range(len(fp))])
    num_mmsi  = np.array([int(f[2]) for f in fp])
    num_mmsi = np.array([num_mmsi[i:].sum() for i in range(len(fp))])
    tot_points = np.array([int(f[3]) for f in fp])
    plt.plot(tot_points,fishing_hours/fishing_hours[0], label = "Fishing Hours")
    plt.plot(tot_points,fishing_hours_adjusted/fishing_hours_adjusted[0], label = "Fishing Hours, \nadding 12 hrs\n    to start and end\nof each segement")
    plt.xlim(0,1000)

    # plt.scatter(tot_points,fishing_hours)
    plt.plot(tot_points,num_mmsi/float(num_mmsi[0]), label="Number of MMSI")
    plt.xlim(0,2000)
    plt.ylim(.2,1.05)
    plt.title("Ratio of fishing hours, mmsi, lost as\n cutoff for number of points increases, {}".format(y))
    plt.xlabel("Vessels with At Least This Many Points per Year")
    plt.ylabel("Ratio of Total")
    plt.legend()
    plt.show()
    
    
    tot_points_.append(tot_points)
    num_mmsi_.append(num_mmsi)
    fishing_hours_adjusted_.append(fishing_hours_adjusted)
    fishing_hours_.append(fishing_hours)

# see how fishing hours vary with number of points

tot_points_active = []
num_mmsi_active = []
fishing_hours_adjusted_active = []
fishing_hours_active = []
for y in years:
    q = '''

    SELECT
      sum(a.fishing_hours) fishing_hours,
      sum(a.fishing_hours_adjusted) fishing_hours_adjusted,
      exact_count_distinct(a.mmsi) num_mmsi,
      b.tot_active_points tot_active_points,
    FROM (
      SELECT
        mmsi,
        SUM(fishing_hours) fishing_hours,
        SUM(fishing_hours_adjusted) fishing_hours_adjusted
      FROM
        [scratch_david.{y}_mmsi_summaries_v2]
      WHERE
        mmsi IN (
        SELECT
          mmsi
        FROM
          scratch_david_mmsi_lists.likely_fishing_{y})
      GROUP BY
        mmsi ) a
    LEFT JOIN (
      SELECT
        mmsi,
        SUM(points) tot_points,
        SUM(active_points) tot_active_points,
        SUM(IF(active_points>=5,1,0)) active_days5,
        SUM(IF(active_points>=10,1,0)) active_days10,
        SUM(IF(active_points>=15,1,0)) active_days15,
        SUM(IF(active_points>=20,1,0)) active_days20,
      FROM
        [scratch_david.{y}_mmsi_summaries_v2]
        group by mmsi) b
    ON
      a.mmsi = b.mmsi
    group by tot_active_points
    order by tot_active_points
    '''.format(y=y)
    fp = Query(q)

    fishing_hours = np.array([float(f[0]) for f in fp])
    fishing_hours = np.array([fishing_hours[i:].sum() for i in range(len(fp))])
    fishing_hours_adjusted = np.array([float(f[1]) for f in fp])
    fishing_hours_adjusted = np.array([fishing_hours_adjusted[i:].sum() for i in range(len(fp))])
    num_mmsi  = np.array([int(f[2]) for f in fp])
    num_mmsi = np.array([num_mmsi[i:].sum() for i in range(len(fp))])
    tot_points = np.array([int(f[3]) for f in fp])
    plt.plot(tot_points,fishing_hours/fishing_hours[0], label = "Fishing Hours")
    plt.plot(tot_points,fishing_hours_adjusted/fishing_hours_adjusted[0], label = "Fishing Hours, \nadding 12 hrs\n    to start and end\nof each segement")
    plt.xlim(0,1000)
    plt.ylim(.2,1.05)

    # plt.scatter(tot_points,fishing_hours)
    plt.plot(tot_points,num_mmsi/float(num_mmsi[0]), label="Number of MMSI")
    plt.xlim(0,2000)
    plt.title("Ratio of fishing hours, mmsi, lost as\n cutoff for number of points increases, {}".format(y))
    plt.xlabel("Vessels with At Least This Many 'Active' Points per Year")
    plt.ylabel("Ratio of Total")
    plt.legend()
    plt.show()
    
    
    tot_points_active.append(tot_points)
    num_mmsi_active.append(num_mmsi)
    fishing_hours_adjusted_active.append(fishing_hours_adjusted)
    fishing_hours_active.append(fishing_hours)



