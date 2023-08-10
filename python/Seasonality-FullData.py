import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt
get_ipython().run_line_magic('matplotlib', 'inline')
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
get_ipython().run_line_magic('matplotlib', 'inline')
# plt.style.use("ggplot")
sns.set_style("darkgrid")

taxi_data = pd.read_csv("../../clean_data/FinalData_for_Models.csv")

taxi_data.shape

taxi_data.rename(columns={'Unnamed: 0':'pickup_time'}, inplace=True)

taxi_data = taxi_data.loc[taxi_data["missing_dt"]==False, :]

taxi_data.shape

taxi_data.head()

# names = cal.holidays(return_name = True)
# data = pd.concat([taxi_data, names], axis=1, join_axes=[taxi_data.tpep_pickup_datetime])

taxi_data.holiday.value_counts(), taxi_data.holiday.value_counts(normalize=True)

count_per_hour = taxi_data.groupby(['holiday','Hour']).num_pickups.sum()

count_df = count_per_hour.unstack(level=0)

count_df["Normal Days"] = count_df[False]/25558
count_df["Federal Holidays"] = count_df[True]/672

count_df.head()

count_df[["Normal Days", "Federal Holidays"]].plot(kind="line")

names = calendar().holidays(return_name = True)

holiday_names = pd.DataFrame(names)

holiday_names=holiday_names.rename(columns={'':'date', '0':'name'}, inplace=True)

holiday_names



