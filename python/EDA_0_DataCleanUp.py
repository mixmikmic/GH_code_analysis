import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

df = pd.read_csv("flight_data.csv")

df.info()
# Columns having NULLs: dep_time, dep_delay, arr_time, arr_delay, tailnum, air_time, 
# These variables require cleanup

# Find count of missing values for each column
df.isnull().sum()

df.head()

# Only dep_delay,arr_delay,air_time,distance are Numerical. 
# Rest are categorical, hence mean etc statistics won't matter for them.
df.loc[:, ["dep_delay","arr_delay","air_time","distance","flight"] ].describe()

plt.figure(figsize=(15,8))
sns.heatmap( round(df.corr(),2), annot=True);

# The only useful takeaway: air_time & distance are 99% correlated. Must be taken into account to estimate air_time

dist_airTime_lookup = df.groupby("distance").air_time.mean().to_frame().round()
dist_airTime_lookup

#Fill air_time based on distance
airTime_missing_idx = df.air_time.isnull()
airTime_toFill = dist_airTime_lookup.loc[df.distance].reset_index()[airTime_missing_idx].air_time

df.loc[airTime_missing_idx, "air_time"] = airTime_toFill

np.sum( df["arr_time"].isnull() & df["dep_time"].notnull() )  # 458: So arr_time can be filled using dep_time. Better to estimate rest of dep_time first.
np.sum( df["arr_time"].notnull() & df["dep_time"].isnull() )  # 0: So dep_time cannot be filled

df_pvtble = df.pivot_table(index="carrier", columns="month", values="dep_delay") 
sns.heatmap(df_pvtble, annot=True)       #carrier-wise average dep_delay for each month
plt.title("carrier-wise average dep_delay for each month");

na_idx = df.dep_delay.isnull()
fill_NA = np.diag( df_pvtble.loc[df.carrier[na_idx], df.month[na_idx]] )
df.loc[na_idx, "dep_delay"] = fill_NA

def add_delay_to_HHMM(HHMM, delay):
    t = (HHMM//100)*60+(HHMM%100) + round(delay)
    t[ t>24*60 ] -= 24*60
    t[ t<0 ] += 24*60
    return (t//60)*100 + t%60

na_idx = df.dep_time.isnull()
df.loc[na_idx, "dep_time"] = add_delay_to_HHMM(df.sched_dep_time, df.dep_delay)[na_idx]

na_idx = df.arr_time.isnull()
df.loc[na_idx, "arr_time"] = add_delay_to_HHMM(df.dep_time, df.air_time)[na_idx]

def HHMM_to_Minutes(HHMM):
    return (HHMM//100) * 60 + (HHMM%100)

def HHMM_diff_inMinutes(x1 , x2, ref):
    """ Perform x1-x2    (both are in HHMM format)
        But what-if date changes? (x1-x2) will be big +ve or big -ve.
        Compare (x1-x2) with "ref" to check whether big is really invalid. If so, do correction."""
    x_diff = HHMM_to_Minutes(x1) - HHMM_to_Minutes(x2)
    # If date changes, this will become large -ve or +ve which is not correct. Correct below
    plt.figure(figsize=(15,3))
    plt.subplot(1,2,1)
    abs(x_diff - ref).hist()    # 23 hrs of deviation found, which is anomaly & indicative of date change
    plt.title("Initial Deviations w.r.t. Reference (absolute)")
    plt.xlabel("Absolute Deviation BINS")
    anomaly_neg = (  (x_diff - ref) < -600  )
    x_diff[anomaly_neg] += 1440

    anomaly_postiv = (    (x_diff - ref) > 600  )
    x_diff[anomaly_postiv] -= 1440

    x_diff.describe()
    plt.subplot(1,2,2)
    abs(x_diff - ref).hist()
    plt.title("Adjusted Deviations w.r.t. Reference (absolute)")
    return x_diff

na_idx = df.arr_delay.isnull()
df.loc[na_idx, "arr_delay"] = HHMM_diff_inMinutes(df.arr_time, df.sched_arr_time, df.dep_delay)[na_idx]

df.isnull().sum()
# Clean-up finished

df.describe()

df.to_csv("flight_data_cleaned.csv")

