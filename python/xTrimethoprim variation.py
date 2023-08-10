#import libraries required for analysis
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import  DateFormatter
import datetime
get_ipython().magic('matplotlib inline')

projectid = "ebmdatalab"
#obtain overall data for prescribing for pregabalin capsules at CCG level
ccg_df = pd.read_gbq("""
SELECT
  trimethoprim.month as period,
  trimethoprim.pct,
  SUM(CASE
      WHEN SUBSTR(trimethoprim.bnf_code,1,9)='0501080W0' THEN items
      ELSE 0 END) AS trimethoprim_items,
  SUM(CASE
      WHEN SUBSTR(trimethoprim.bnf_code,1,9)='0501130R0' THEN items
      ELSE 0 END) AS nitrofurantoin_items,
  SUM(items) AS all_items,
  IEEE_DIVIDE(SUM(
      CASE
      WHEN SUBSTR(trimethoprim.bnf_code,1,9)='0501080W0' THEN items
        ELSE 0 END), SUM(items)) AS trimethoprim_percent
FROM
ebmdatalab.richard.trimethoprim_data AS trimethoprim
INNER JOIN
  ebmdatalab.hscic.ccgs AS ccg
ON
  trimethoprim.pct=ccg.code
WHERE
 ccg.org_type = 'CCG'
GROUP BY
  period,
  pct
ORDER BY
  period,
  pct
""", projectid, dialect='standard')

projectid = "ebmdatalab"
#obtain overall data for prescribing for pregabalin capsules at practice level
practice_df = pd.read_gbq("""
SELECT
  trimethoprim.month as period,
  trimethoprim.practice,
  SUM(CASE
      WHEN SUBSTR(trimethoprim.bnf_code,1,9)='0501080W0' THEN items
      ELSE 0 END) AS trimethoprim_items,
  SUM(CASE
      WHEN SUBSTR(trimethoprim.bnf_code,1,9)='0501130R0' THEN items
      ELSE 0 END) AS nitrofurantoin_items,
  SUM(items) AS all_items,
  IEEE_DIVIDE(SUM(
      CASE
      WHEN SUBSTR(trimethoprim.bnf_code,1,9)='0501080W0' THEN items
        ELSE 0 END), SUM(items)) AS trimethoprim_percent
FROM
  ebmdatalab.richard.trimethoprim_data AS trimethoprim
INNER JOIN
  ebmdatalab.hscic.practice_statistics AS listsize
ON
  trimethoprim.practice=listsize.practice
AND   
  trimethoprim.month=listsize.month
WHERE
  listsize.total_list_size >=1000
GROUP BY
  period,
  practice
ORDER BY
  period,
  practice
""", projectid, dialect='standard')

#create deciles for practices
x = np.arange(0.1, 1, 0.1)
practice_deciles = practice_df.groupby('period')['trimethoprim_percent'].quantile(x)
practice_deciles_df=pd.DataFrame(practice_deciles)
practice_deciles_df=practice_deciles_df.reset_index()
# create integer range of percentiles as integers are better for  charts
practice_deciles_df["index"] = (practice_deciles_df.level_1*10).map(int)
practice_deciles_df['period'] = practice_deciles_df['period'].astype(str)
# set format for dates:
practice_deciles_df['period'] = [datetime.datetime.strptime(date, '%Y-%m-%d').date() for date in practice_deciles_df['period']]

#create deciles for CCGs
x = np.arange(0.1, 1, 0.1)
ccg_deciles = ccg_df.groupby('period')['trimethoprim_percent'].quantile(x)
ccg_deciles_df=pd.DataFrame(ccg_deciles)
ccg_deciles_df=ccg_deciles_df.reset_index()
# create integer range of percentiles as integers are better for  charts
ccg_deciles_df["index"] = (ccg_deciles_df.level_1*10).map(int)
ccg_deciles_df['period'] = ccg_deciles_df['period'].astype(str)
# set format for dates:
ccg_deciles_df['period'] = [datetime.datetime.strptime(date, '%Y-%m-%d').date() for date in ccg_deciles_df['period']]

# Plot time series charts of deciles
import matplotlib.pyplot as plt
import datetime
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set_style("whitegrid",{'grid.color': '.9'})

fig = plt.figure(figsize=(16,6.666)) 
gs = gridspec.GridSpec(1,2)  # grid layout for subplots

# set sort order of drugs manually, and add grid refs to position each subplot:
s = [(0,ccg_deciles_df,0,0,'CCGs'), (1,practice_deciles_df,0,1,'practices')]

# Plot each subplot using a loop
for i in s:
    ax = plt.subplot(gs[i[2], i[3]])  # position of subplot in grid using coordinates listed in s
    for decile in range(1,10):   # plot each decile line
        data = i[1].loc[(i[1]['index']==decile)]
        if decile == 5:
            ax.plot(data["period"],100*data['trimethoprim_percent'],'b-',linewidth=0.7)
        else:
            ax.plot(data["period"],100*data['trimethoprim_percent'],'b--',linewidth=0.4)
    if  i[3]%2==0:    # set y axis title only for charts in leftmost column
        ax.set_ylabel('Trimethoprim as percentage of trimethoprim & nitrofurantoin items', size =15, alpha=0.6)
    ax.set_title(i[4],size = 18)
    ax.set_ylim([0, 100*i[1]['trimethoprim_percent'].max()*1.05])  # set ymax across all subplots as largest value across dataset
    ax.tick_params(labelsize=12)
    ax.set_xlim([i[1]['period'].min(), i[1]['period'].max()]) # set x axis range as full date range
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%B %Y'))
    
plt.subplots_adjust(wspace = 0.07,hspace = 0.15)
#plt.savefig('Figure X.png', format='png', dpi=300)
plt.show()



