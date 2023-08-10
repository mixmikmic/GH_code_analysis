import pandas as pd
import seaborn as sns
sns.set_context('poster', font_scale=1.25)

get_ipython().run_line_magic('pylab', 'inline')

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2

from current_application import get_sql_dataset
from current_application import prepare_dataset

con = get_sql_dataset.connect_to_sql()
con = psycopg2.connect(database='mimic', user='nwespe', host='localhost')
cur = con.cursor()
cur.execute('SET search_path to mimiciii;')

sql_query = """
SELECT * FROM all_admit_info;"""

all_patients = pd.read_sql_query(sql_query, con)

sql_query = """
SELECT me.hadm_id, me.org_name, me.charttime, 
ai.admittime, ai.dischtime, ai.hospital_expire_flag
, ROUND( (CAST(EXTRACT(epoch FROM me.charttime - ai.admittime)/(60*60*24) AS numeric)), 4) AS cdiff_timelag
, ROUND( (CAST(EXTRACT(epoch FROM ai.dischtime - ai.admittime)/(60*60*24) AS numeric)), 4) AS los
FROM microbiologyevents me
JOIN admit_info ai ON ai.hadm_id = me.hadm_id
WHERE me.org_name ILIKE '%DIFF%';"""

cdiff_patients = pd.read_sql_query(sql_query, con)

cdiff_patients.head()

len(cdiff_patients)

cdiff_patients.rename(columns={'hospital_expire_flag':'expire'}, inplace=True)

los_survived = cdiff_patients[cdiff_patients.expire == 0].los.dropna()
los_expired = cdiff_patients[cdiff_patients.expire == 1].los.dropna()
#bins=np.linspace(0, 20, 8)
fig, ax = plt.subplots(1, 2)
bins = np.linspace(0, 50, 10)
ax[0].hist(los_survived, color='b', bins=bins)
ax[1].hist(los_expired, color='r', bins=bins)

cdiff_ids = list(cdiff_patients.hadm_id.values)
noncd_patients = all_patients[-all_patients.hadm_id.isin(cdiff_ids)]

noncd_patients.head()

g = sns.FacetGrid(cdiff_patients, col="expire", margin_titles=True, sharey=False)
#bins = np.linspace(0, 50, 10)
g.map(sns.kdeplot, "los") #

g = sns.FacetGrid(noncd_patients, col="hospital_expire_flag", margin_titles=True, sharey=False)
#bins = np.linspace(0, 50, 10)
g.map(sns.kdeplot, "los_hospital") #, bins=bins, lw=0

def check_id(x):
    if x in cdiff_ids:
        return 1
    else:
        return 0

all_patients['cdiff'] = all_patients.apply(lambda x: check_id(x['hadm_id']), axis=1)

len(all_patients)

all_patients.head()

g = sns.FacetGrid(all_patients, col="hospital_expire_flag", hue='cdiff', 
                  margin_titles=True, sharey=False)
#bins = np.linspace(0, 50, 10)
g.map(sns.kdeplot, "los_hospital", shade=True, clip=(0,90)) #, bins=bins, lw=0

noncdiff_mortality = float(len(all_patients[(all_patients.hospital_expire_flag == 1) &
                                      (all_patients.cdiff == 0)]))/len(all_patients[all_patients.cdiff == 0])

cdiff_mortality = float(len(all_patients[(all_patients.hospital_expire_flag == 1) &
                                      (all_patients.cdiff == 1)]))/len(all_patients[all_patients.cdiff == 1])

noncdiff_mortality, cdiff_mortality

sns.set_context('poster', font_scale=1.25)

fig, ax  = plt.subplots(1, 2, figsize=(10, 3))  # #114C81, #34A5DA, #18679A
fig.subplots_adjust(wspace=.5)
fig.tight_layout()
ax[0].barh([1, 2], [19.2, 22.5], color=['#11426E','#AA1728'], height=0.5,
       tick_label=['Non-C. diff', 'C. diff'], align='center')
ax[0].set_xlim(0, 30)
ax[0].set_xlabel('Mortality (%)')

ax[1].barh([1, 2], [11.8, 16.5], color=['#11426E','#AA1728'], height=0.5,
       tick_label=['Non-C. diff', 'C. diff'], align='center')
ax[1].set_xlim(0, 20)
ax[1].set_xticks([0, 5, 10, 15, 20])
ax[1].set_xlabel('Length of stay (days)')

plt.savefig('/Users/nwespe/Desktop/cdiff_stats.svg', bbox_inches='tight')

avg_los = all_patients.groupby(['cdiff', 'hospital_expire_flag'], axis=0)['los_hospital'].describe()

avg_los

all_features, cdiff_data, outcomes = get_sql_dataset.main()

cdiff_data.head()

noncdiff_mortality = float(len(cdiff_data[(cdiff_data.expire == 1) &
                                      (cdiff_data.outcome == 0)]))/len(cdiff_data[cdiff_data.outcome == 0])

cdiff_mortality = float(len(cdiff_data[(cdiff_data.expire == 1) &
                                      (cdiff_data.outcome == 1)]))/len(cdiff_data[cdiff_data.outcome == 1])

noncdiff_mortality, cdiff_mortality

(14.2-9.6)/9.6



fig, ax  = plt.subplots(1, 2, figsize=(10, 3))  # #114C81, #34A5DA, #18679A
fig.subplots_adjust(wspace=.5)
fig.tight_layout()
ax[0].barh([1, 2], [9.6, 14.2], color=['#34A5DA','#F96928'], height=0.5,
       tick_label=['Non-C. diff', 'C. diff'], align='center')
ax[0].set_xlim(0, 15)
ax[0].set_xlabel('Mortality (%)')

ax[1].barh([1, 2], [11.8, 19.5], color=['#34A5DA','#F96928'], height=0.5,
       tick_label=['Non-C. diff', 'C. diff'], align='center')
ax[1].set_xlim(0, 20)
ax[1].set_xticks([0, 5, 10, 15, 20])
ax[1].set_xlabel('Length of stay (days)')

plt.savefig('/Users/nwespe/Desktop/cdiff_stats.svg', bbox_inches='tight')



avg_los = cdiff_data.groupby(['outcome', 'expire'], axis=0)['los_hospital'].describe()

avg_los

(19.8-11.8)/11.8

cdiff_data.columns

import itertools

means = cdiff_data.groupby(['outcome'], axis=0)['admittime'].mean()
g = sns.FacetGrid(cdiff_data, hue='outcome', margin_titles=True, sharey=False)
g.map(sns.kdeplot, 'admittime', shade=True) #, bins=bins, lw=0, clip=(0,90)
axes = plt.gca()
#axes.set_ylim([0, 0.03])
for m in means:
    plt.axvline(m, linestyle='dashed', linewidth=2)
plt.savefig('/Users/nwespe/Desktop/Week4Graphics/Plots/admittime.svg', bbox_inches='tight')

ncols = 6
nrows = len(cdiff_data.columns)//6 + 1
axis_ids = list(itertools.product(xrange(nrows), xrange(ncols)))


#bins = np.linspace(0, 50, 10)
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*4))
fig.tight_layout()
fig.subplots_adjust(wspace=0.5, hspace=0.5)
for ix, c in enumerate(cdiff_data.columns):
    i, j = axis_ids[ix]
    g = sns.FacetGrid(cdiff_data, hue='outcome', margin_titles=True, sharey=False)
    g.map(sns.kdeplot, c, shade=True, ax=axes[i, j]) #, bins=bins, lw=0, clip=(0,90)
    axes[i, j].set_title(c)



